package org.firstinspires.ftc.teamcode.vision;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.highgui.HighGui;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Real-time test for VisionLocalizer.
 * Connect your Arducam and run this on your PC to see live localization.
 */
public class VisionLocalizerRealtimeTest {

    static {
        // Automatically detect architecture and load the correct native library
        String arch = System.getProperty("os.arch").toLowerCase();
        String libSuffix = arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64";
        String libPath = System.getProperty("user.dir") + "/TeamCode/libs/desktop/libopencv_java490" + (arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64") + (System.getProperty("os.name").toLowerCase().contains("mac") ? ".dylib" : ".so");
        
        System.out.println("Loading native library from: " + libPath);
        System.load(libPath);
    }

    public static void main(String[] args) {
        // 1. Setup Localizer with Calibrated Parameters from VisionLocalizer
        Map<Integer, VisionLocalizer.VisionPose3D> fieldMap = VisionLocalizer.DECODE_FIELD_MAP;
        
        Mat cameraMatrix = VisionLocalizer.getArducamMatrix();
        MatOfDouble distCoeffs = VisionLocalizer.getArducamDistCoeffs();
        
        // Use standard camera transform from VisionLocalizer
        VisionLocalizer.VisionPose3D cameraTransform = VisionLocalizer.ARDUCAM_TRANSFORM;
        
        VisionLocalizer localizer = new VisionLocalizer(fieldMap, cameraMatrix, distCoeffs, cameraTransform);

        // 2. Setup AprilTag Detector (ArUco Module)
        Dictionary dictionary = Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11);
        ArucoDetector detector = new ArucoDetector(dictionary);

        // 3. Initialize Camera
        VideoCapture cap = new VideoCapture(0); // Arducam is at index 0
        if (!cap.isOpened()) {
            System.err.println("Error: Could not open camera!");
            return;
        }
        
        // Set capture resolution to 1920x1080
        cap.set(Videoio.CAP_PROP_FRAME_WIDTH, 1920); 
        cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, 1080);

        Mat frame = new Mat();
        Mat resized = new Mat();
        
        System.out.println("Real-time Vision Started.");
        System.out.println("QUIT OPTIONS:");
        System.out.println("1. Press 'q' or ESC in the OpenCV window.");
        System.out.println("2. Press ENTER here in the terminal.");

        // Terminal Listener Thread
        final boolean[] shouldQuit = {false};
        Thread terminalListener = new Thread(() -> {
            try {
                System.in.read();
                shouldQuit[0] = true;
                System.out.println("Terminal exit signal received.");
            } catch (Exception e) {}
        });
        terminalListener.setDaemon(true);
        terminalListener.start();

        while (!shouldQuit[0]) {
            if (cap.read(frame)) {
                // Resize to 1280x720 using the shared logic in Localizer
                localizer.prepareFrame(frame, resized);

                // Detect Tags
                List<Mat> corners = new ArrayList<>();
                Mat ids = new Mat();
                detector.detectMarkers(resized, corners, ids);

                List<VisionLocalizer.Detection> detections = new ArrayList<>();
                if (!ids.empty()) {
                    for (int i = 0; i < ids.rows(); i++) {
                        int id = (int) ids.get(i, 0)[0];
                        Mat cornerMat = corners.get(i); // 1x4 matrix of corners
                        
                        Point[] pts = new Point[4];
                        for (int j = 0; j < 4; j++) {
                            pts[j] = new Point(cornerMat.get(0, j)[0], cornerMat.get(0, j)[1]);
                        }
                        detections.add(new VisionLocalizer.Detection(id, pts));
                        
                        // Draw outline on image
                        for (int j = 0; j < 4; j++) {
                            Imgproc.line(resized, pts[j], pts[(j + 1) % 4], new Scalar(0, 255, 0), 2);
                        }
                        Imgproc.putText(resized, "ID: " + id, pts[0], Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 255, 0), 2);
                    }
                }

                // Estimate Pose (Assuming IMU Yaw = 0 for PC test, or you can simulate rotation)
                double simulatedImuYaw = 0; 
                VisionLocalizer.VisionPose3D robotPose = localizer.estimateRobotPose(detections, simulatedImuYaw);

                // Overlay Info
                if (robotPose != null) {
                    String poseText = String.format("Robot Pose: X=%.1f, Y=%.1f, Z=%.1f", robotPose.x, robotPose.y, robotPose.z);
                    Imgproc.putText(resized, poseText, new Point(30, 50), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 3);
                } else {
                    Imgproc.putText(resized, "No Tags Detected", new Point(30, 50), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255), 3);
                }

                // Show Window
                HighGui.imshow("VisionLocalizer Real-time Test", resized);
                
                // Exit on 'q' or ESC
                int key = HighGui.waitKey(30);
                if (key >= 0) {
                    System.out.println("Key pressed: " + key);
                    // 113 is 'q', 81 is 'Q', 27 is 'ESC'
                    if (key == 113 || key == 81 || key == 27) {
                        System.out.println("Exit signal received.");
                        break;
                    }
                }
            }
        }

        System.out.println("Closing camera and windows...");
        cap.release();
        HighGui.destroyAllWindows();
        System.out.println("Done.");
        System.exit(0);
    }
}
