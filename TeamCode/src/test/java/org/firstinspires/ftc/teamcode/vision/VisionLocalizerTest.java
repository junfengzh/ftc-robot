package org.firstinspires.ftc.teamcode.vision;

import org.opencv.core.*;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

/**
 * PC Test for VisionLocalizer. 
 * Run this on your laptop to verify the localization math without a robot.
 */
public class VisionLocalizerTest {

    static {
        // Automatically detect architecture and load the correct native library
        String arch = System.getProperty("os.arch").toLowerCase();
        String libSuffix = arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64";
        String libPath = System.getProperty("user.dir") + "/TeamCode/libs/desktop/libopencv_java490" + (arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64") + (System.getProperty("os.name").toLowerCase().contains("mac") ? ".dylib" : ".so");
        
        System.out.println("Loading native library for " + arch + " from: " + libPath);
        System.load(libPath);
    }

    public static void main(String[] args) {
        System.out.println("Starting VisionLocalizer Test with DECODE Field Map...");

        // 1. Setup Field Map from official DECODE constants
        Map<Integer, VisionLocalizer.VisionPose3D> fieldMap = VisionLocalizer.DECODE_FIELD_MAP;

        // 2. Setup Camera Parameters (Calibrated Arducam at 1280x720)
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, 
            518.5498, 0.0, 627.7407,
            0.0, 516.8600, 375.1764,
            0.0, 0.0, 1.0);
        
        MatOfDouble distCoeffs = VisionLocalizer.getArducamDistCoeffs();

        // Use standard camera transform from VisionLocalizer
        VisionLocalizer.VisionPose3D cameraTransform = VisionLocalizer.ARDUCAM_TRANSFORM;

        VisionLocalizer localizer = new VisionLocalizer(fieldMap, cameraMatrix, distCoeffs, cameraTransform);

        // 3. Simulate Detection (Tag 20 - Blue Goal at center of frame for 1280x720)
        List<VisionLocalizer.Detection> detections = new ArrayList<>();
        // Center is roughly (640, 360)
        Point[] corners = {
            new Point(590, 310), new Point(690, 310), 
            new Point(690, 410), new Point(590, 410)
        };
        detections.add(new VisionLocalizer.Detection(20, corners));

        // 4. Estimate Pose
        double imuYaw = Math.toRadians(0); // Facing straight ahead (Field Relative)
        VisionLocalizer.VisionPose3D robotPose = localizer.estimateRobotPose(detections, imuYaw);

        if (robotPose != null) {
            System.out.println("Robot Position Estimated Successfully!");
            System.out.println(robotPose.toString());
        } else {
            System.out.println("Failed to estimate robot position.");
        }
    }
}
