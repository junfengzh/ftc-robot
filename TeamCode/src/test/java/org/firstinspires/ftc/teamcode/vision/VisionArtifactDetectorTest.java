package org.firstinspires.ftc.teamcode.vision;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.List;

/**
 * PC Test for VisionArtifactDetector.
 * Verify ground coordinate calculations and segmentation.
 */
public class VisionArtifactDetectorTest {

    static {
        // Automatically detect architecture and load the correct native library
        String arch = System.getProperty("os.arch").toLowerCase();
        String libSuffix = arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64";
        String libPath = System.getProperty("user.dir") + "/TeamCode/libs/desktop/libopencv_java490" + (arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64") + (System.getProperty("os.name").toLowerCase().contains("mac") ? ".dylib" : ".so");
        
        System.out.println("Loading native library for " + arch + " from: " + libPath);
        System.load(libPath);
    }

    public static void main(String[] args) {
        System.out.println("Starting VisionArtifactDetector Test...");

        // 1. Setup Mock Camera Matrix (Logitech C920 at 640x480)
        Mat cameraMatrix = Mat.eye(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, 600, 0, 320, 0, 600, 240, 0, 0, 1);
        Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);

        // 2. Setup Homography Matrix (Pixel -> Ground in inches)
        // (Assume 1 inch = 10 pixels for this test)
        Mat homography = Mat.eye(3, 3, CvType.CV_64F);
        homography.put(0, 0, 0.1, 0, -32, 0, 0.1, -24, 0, 0, 1);

        VisionArtifactDetector detector = new VisionArtifactDetector(cameraMatrix, distCoeffs, homography);
        
        // Setup Yellow Detection (for Sample)
        detector.setHSVRange(new Scalar(20, 100, 100), new Scalar(30, 255, 255));

        // 3. Create a Mock Frame (Black image with a Yellow square)
        Mat frame = Mat.zeros(480, 640, CvType.CV_8UC3);
        // Draw a yellow rectangle (in BGR: Blue=0, Green=255, Red=255)
        Imgproc.rectangle(frame, new Point(300, 220), new Point(340, 260), new Scalar(0, 255, 255), -1);

        // 4. Run Detection
        List<VisionArtifactDetector.Artifact> artifacts = detector.detect(frame);

        if (!artifacts.isEmpty()) {
            System.out.println("Artifacts Detected Successfully!");
            for (VisionArtifactDetector.Artifact a : artifacts) {
                System.out.println(a.toString());
            }
        } else {
            System.out.println("No artifacts detected in the mock frame.");
        }
    }
}
