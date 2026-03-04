package org.firstinspires.ftc.teamcode.vision;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * VisionLocalizer implements the MegaTag 2 algorithm for robot localization using AprilTags.
 */
public class VisionLocalizer {

    // Arducam Calibration Constants (1280x720 resolution)
    public static final double FX = 518.5498;
    public static final double FY = 516.8600;
    public static final double CX = 627.7407;
    public static final double CY = 375.1764;

    public static final double[] DIST_COEFFS = {
        -0.285759, 0.079079, -0.000350, 0.000320, -0.009423
    };

    // Default Camera Transform (relative to robot center at floor level)
    // X=-5" (Backward), Y=-5" (Right), Z=15" (Above Floor)
    public static final VisionPose3D ARDUCAM_TRANSFORM = new VisionPose3D(-5.0, -5.0, 15.0, 0.0, 0.0, 0.0);

    // DECODE Season Field Constants (2025-2026) - Z is inches above floor
    public static final double TAG_SIZE_INCHES = 6.5; 
    public static final Map<Integer, VisionPose3D> DECODE_FIELD_MAP = new HashMap<Integer, VisionPose3D>() {{
        put(20, new VisionPose3D(-58.37, -55.64, 29.50, 0.0, 0.0, Math.toRadians(54.0)));
        put(24, new VisionPose3D(-58.37, 55.64, 29.50, 0.0, 0.0, Math.toRadians(-54.0)));
    }};

    private final Map<Integer, VisionPose3D> fieldMap;
    private final Mat cameraMatrix;
    private final MatOfDouble distCoeffs;
    private final VisionPose3D cameraTransform;

    public VisionLocalizer(Map<Integer, VisionPose3D> fieldMap, Mat cameraMatrix, MatOfDouble distCoeffs, 
                          VisionPose3D cameraTransform) {
        this.fieldMap = fieldMap;
        this.cameraMatrix = cameraMatrix;
        this.distCoeffs = distCoeffs;
        this.cameraTransform = cameraTransform;
    }

    public static Mat getArducamMatrix() {
        Mat matrix = new Mat(3, 3, CvType.CV_64F);
        matrix.put(0, 0, FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0);
        return matrix;
    }

    public static MatOfDouble getArducamDistCoeffs() {
        return new MatOfDouble(DIST_COEFFS);
    }

    public void prepareFrame(Mat input, Mat output) {
        Imgproc.resize(input, output, new Size(1280, 720));
    }

    public VisionPose3D estimateRobotPose(List<Detection> detections, double imuYawRadians) {
        if (detections == null || detections.isEmpty()) return null;

        List<VisionPose3D> robotPoses = new ArrayList<>();

        for (Detection detection : detections) {
            if (!fieldMap.containsKey(detection.id)) continue;

            VisionPose3D tagFieldPose = fieldMap.get(detection.id);
            double tagSize = TAG_SIZE_INCHES;
            
            // Standard OpenCV Y-Down convention
            MatOfPoint3f objectPoints = new MatOfPoint3f(
                new Point3(-tagSize/2, -tagSize/2, 0), // TL
                new Point3( tagSize/2, -tagSize/2, 0), // TR
                new Point3( tagSize/2,  tagSize/2, 0), // BR
                new Point3(-tagSize/2,  tagSize/2, 0)  // BL
            );

            MatOfPoint2f imagePoints = new MatOfPoint2f(detection.corners);
            Mat rvec = new Mat();
            Mat tvec = new Mat();
            
            if (Calib3d.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec)) {
                // PURE GEOMETRIC TRANSFORM
                double tx = tvec.get(0, 0)[0];
                double ty = tvec.get(1, 0)[0];
                double tz = tvec.get(2, 0)[0];

                // 1. Map Camera-Local to Robot-Local (Forward-Left-Up)
                double tagRelCamForward = tz;
                double tagRelCamLeft = -tx;
                double tagRelCamUp = -ty;

                // 2. Vector Robot -> Tag (Robot Frame)
                double tagRelRobotForward = tagRelCamForward + cameraTransform.x;
                double tagRelRobotLeft = tagRelCamLeft + cameraTransform.y;
                double tagRelRobotUp = tagRelCamUp + cameraTransform.z;

                // 3. Rotate Tag-relative-to-Robot into Field-Frame vector
                double cosYaw = Math.cos(imuYawRadians);
                double sinYaw = Math.sin(imuYawRadians);
                
                double tagRelFieldX = tagRelRobotForward * cosYaw - tagRelRobotLeft * sinYaw;
                double tagRelFieldY = tagRelRobotForward * sinYaw + tagRelRobotLeft * cosYaw;
                double tagRelFieldZ = tagRelRobotUp;

                // 4. Calculate Robot Position
                double robotX = tagFieldPose.x - tagRelFieldX;
                double robotY = tagFieldPose.y - tagRelFieldY;
                double robotZ = tagFieldPose.z - tagRelFieldZ;

                robotPoses.add(new VisionPose3D(robotX, robotY, robotZ, 0, 0, imuYawRadians));
            }
            rvec.release(); tvec.release(); objectPoints.release(); imagePoints.release();
        }

        if (robotPoses.isEmpty()) return null;

        double avgX = 0, avgY = 0, avgZ = 0;
        for (VisionPose3D p : robotPoses) {
            avgX += p.x; avgY += p.y; avgZ += p.z;
        }
        return new VisionPose3D(avgX / robotPoses.size(), avgY / robotPoses.size(), avgZ / robotPoses.size(), 0, 0, imuYawRadians);
    }

    private double[][] getRotationMatrix(double roll, double pitch, double yaw) {
        double cr = Math.cos(roll), sr = Math.sin(roll);
        double cp = Math.cos(pitch), sp = Math.sin(pitch);
        double cy = Math.cos(yaw), sy = Math.sin(yaw);
        double[][] R = new double[3][3];
        R[0][0] = cy * cp;
        R[0][1] = cy * sp * sr - sy * cr;
        R[0][2] = cy * sp * cr + sy * sr;
        R[1][0] = sy * cp;
        R[1][1] = sy * sp * sr + cy * cr;
        R[1][2] = sy * sp * cr - cy * sr;
        R[2][0] = -sp;
        R[2][1] = cp * sr;
        R[2][2] = cp * cr;
        return R;
    }

    private double[] multiply(double[][] matrix, double[] vector) {
        double[] result = new double[3];
        for (int i = 0; i < 3; i++) result[i] = matrix[i][0] * vector[0] + matrix[i][1] * vector[1] + matrix[i][2] * vector[2];
        return result;
    }

    public static class VisionPose3D {
        public double x, y, z, roll, pitch, yaw;
        public VisionPose3D(double x, double y, double z, double roll, double pitch, double yaw) {
            this.x = x; this.y = y; this.z = z; this.roll = roll; this.pitch = pitch; this.yaw = yaw;
        }
        public String toString() { return String.format("Pose3D(x=%.2f, y=%.2f, z=%.2f, yaw=%.2f°)", x, y, z, Math.toDegrees(yaw)); }
    }

    public static class Detection {
        public int id; public Point[] corners;
        public Detection(int id, Point[] corners) { this.id = id; this.corners = corners; }
    }
}
