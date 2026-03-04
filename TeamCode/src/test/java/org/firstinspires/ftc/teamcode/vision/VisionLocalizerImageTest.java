package org.firstinspires.ftc.teamcode.vision;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class VisionLocalizerImageTest {

    static {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        String libSuffix = arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64";
        String ext = os.contains("mac") ? ".dylib" : ".so";
        
        String libPath = System.getProperty("user.dir") + "/TeamCode/libs/desktop/libopencv_java490" + libSuffix + ext;
        System.out.println("Loading " + os + " (" + arch + ") native library from: " + libPath);
        
        try {
            System.load(libPath);
            System.out.println("Successfully loaded OpenCV native library.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load OpenCV native library at " + libPath);
            throw e;
        }
    }

    static class TestCase {
        String name;
        String path;
        double gtX, gtY;
        double simulatedYaw;

        TestCase(String name, String path, double gtX, double gtY, double simulatedYaw) {
            this.name = name;
            this.path = path;
            this.gtX = gtX;
            this.gtY = gtY;
            this.simulatedYaw = simulatedYaw;
        }
    }

    public static void main(String[] args) {
        String dataDir = System.getProperty("user.dir") + "/data/vision_localizer/";
        
        List<TestCase> tests = new ArrayList<>();
        // Current Best Guess: Robot at X=64, Y=0 facing wall (180 deg)
        tests.add(new TestCase("Arducam 1", dataDir + "arducam1.jpg", 64.0, 0.0, 180.0));
        // Arducam 1_1758: Robot at -45, 45 facing wall (310 deg)
        tests.add(new TestCase("Arducam 1_1758", dataDir + "arducam1_1758.jpg", -45.0, 45.0, 310.0));

        // Setup Localizer
        Map<Integer, VisionLocalizer.VisionPose3D> fieldMap = VisionLocalizer.DECODE_FIELD_MAP;
        Mat cameraMatrix = VisionLocalizer.getArducamMatrix();
        MatOfDouble distCoeffs = VisionLocalizer.getArducamDistCoeffs();
        VisionLocalizer.VisionPose3D cameraTransform = VisionLocalizer.ARDUCAM_TRANSFORM;
        VisionLocalizer localizer = new VisionLocalizer(fieldMap, cameraMatrix, distCoeffs, cameraTransform);

        // Setup Detector
        Dictionary dictionary = Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11);
        ArucoDetector detector = new ArucoDetector(dictionary);

        System.out.println("=== VisionLocalizer Accuracy Test ===");
        System.out.println("Using Tag Size: " + VisionLocalizer.TAG_SIZE_INCHES + " inches");

        for (TestCase test : tests) {
            System.out.println("\n--- Testing: " + test.name + " ---");
            Mat frame = Imgcodecs.imread(test.path);
            if (frame.empty()) {
                System.err.println("Could not load: " + test.path);
                continue;
            }

            Mat resized = new Mat();
            localizer.prepareFrame(frame, resized);

            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            detector.detectMarkers(resized, corners, ids);

            List<VisionLocalizer.Detection> detections = new ArrayList<>();
            if (!ids.empty()) {
                for (int i = 0; i < ids.rows(); i++) {
                    int id = (int) ids.get(i, 0)[0];
                    Mat cornerMat = corners.get(i);
                    Point[] pts = new Point[4];
                    for (int j = 0; j < 4; j++) {
                        pts[j] = new Point(cornerMat.get(0, j)[0], cornerMat.get(0, j)[1]);
                    }
                    detections.add(new VisionLocalizer.Detection(id, pts));
                    System.out.print("ID " + id + " ");
                }
                System.out.println();
            }

            double imuYaw = Math.toRadians(test.simulatedYaw);
            VisionLocalizer.VisionPose3D pose = localizer.estimateRobotPose(detections, imuYaw);

            if (pose != null) {
                System.out.println("Calculated Pose: " + pose.toString());
                System.out.printf("Ground Truth: X=%.2f, Y=%.2f\n", test.gtX, test.gtY);
                
                double errX = pose.x - test.gtX;
                double errY = pose.y - test.gtY;
                System.out.printf("Error: dX=%.2f, dY=%.2f, Dist=%.2f\n", errX, errY, Math.sqrt(errX*errX + errY*errY));

                if (test.name.equals("Arducam 1_1758")) {
                    // Hypothesis: What if X is 45.0 and Y is -45.0?
                    double hypX = 45.0, hypY = -45.0;
                    double hErrX = pose.x - hypX, hErrY = pose.y - hypY;
                    System.out.printf("Hypothesis (X=45, Y=-45): dX=%.2f, dY=%.2f, Dist=%.2f\n", hErrX, hErrY, Math.sqrt(hErrX*hErrX + hErrY*hErrY));
                }
            } else {
                System.out.println("Pose estimation FAILED.");
            }
        }
        System.out.println("\nDone.");
        System.exit(0);
    }
}
