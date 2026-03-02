package org.firstinspires.ftc.teamcode.vision;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;

import java.util.ArrayList;
import java.util.List;

/**
 * VisionArtifactDetector identifies game elements (artifacts) using color segmentation
 * and projects their coordinates to the ground plane using Homography.
 */
public class VisionArtifactDetector {

    public static class Artifact {
        public String type; // e.g. "Sample", "Yellow", "Blue", "Red"
        public double x, y; // Ground coordinates relative to camera (e.g., in inches or mm)
        public double angle; // Orientation of the artifact

        public Artifact(String type, double x, double y, double angle) {
            this.type = type; this.x = x; this.y = y; this.angle = angle;
        }

        @Override
        public String toString() {
            return String.format("%s at (x=%.2f, y=%.2f) angle=%.1f°", type, x, y, angle);
        }
    }

    private final Mat cameraMatrix;
    private final Mat distCoeffs;
    private final Mat homographyMatrix;

    // Detection Parameters
    private Scalar lowerHSV = new Scalar(0, 50, 50);  // Default: Red-ish
    private Scalar upperHSV = new Scalar(20, 255, 255);

    public VisionArtifactDetector(Mat cameraMatrix, Mat distCoeffs, Mat homographyMatrix) {
        this.cameraMatrix = cameraMatrix;
        this.distCoeffs = distCoeffs;
        this.homographyMatrix = homographyMatrix;
    }

    public void setHSVRange(Scalar lower, Scalar upper) {
        this.lowerHSV = lower;
        this.upperHSV = upper;
    }

    /**
     * Processes a single frame to detect artifacts.
     * 
     * @param frame Raw camera frame (BGR/RGB)
     * @return List of detected artifacts with ground coordinates
     */
    public List<Artifact> detect(Mat frame) {
        List<Artifact> results = new ArrayList<>();
        if (frame == null || frame.empty()) return results;

        // 1. Color Segmentation
        Mat hsv = new Mat();
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
        
        Mat mask = new Mat();
        Core.inRange(hsv, lowerHSV, upperHSV, mask);
        
        // Morphological operations to clean up noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

        // 2. Contour Analysis
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < 500) continue; // Filter small noise

            // Get oriented bounding box
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            RotatedRect rotatedRect = Imgproc.minAreaRect(contour2f);
            
            // 3. Coordinate Transformation (Pixel -> Ground)
            Point pixelPoint = rotatedRect.center;
            Point groundPoint = projectToGround(pixelPoint);

            results.add(new Artifact("Target", groundPoint.x, groundPoint.y, rotatedRect.angle));
            
            contour2f.release();
        }

        // Cleanup
        hsv.release();
        mask.release();
        hierarchy.release();
        kernel.release();
        for (MatOfPoint c : contours) c.release();

        return results;
    }

    /**
     * Projects a pixel coordinate to the ground plane using the homography matrix.
     * 
     * @param pixelPoint Point in image coordinates (u, v)
     * @return Point in ground coordinates (x, y)
     */
    private Point projectToGround(Point pixelPoint) {
        // [u, v, 1] vector
        Mat src = new Mat(3, 1, CvType.CV_64F);
        src.put(0, 0, pixelPoint.x);
        src.put(1, 0, pixelPoint.y);
        src.put(2, 0, 1.0);

        // Ground = H * Pixel
        Mat dst = new Mat(3, 1, CvType.CV_64F);
        Core.gemm(homographyMatrix, src, 1, new Mat(), 0, dst);

        // Normalize w-coordinate (Perspective divide)
        double w = dst.get(2, 0)[0];
        Point ground = new Point(dst.get(0, 0)[0] / w, dst.get(1, 0)[0] / w);

        src.release();
        dst.release();
        return ground;
    }
}
