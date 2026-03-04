package org.firstinspires.ftc.teamcode.vision;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class VisionLocalizerImageSequenceTest {

    static {
        String arch = System.getProperty("os.arch").toLowerCase();
        String libSuffix = arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64";
        String libPath = System.getProperty("user.dir") + "/TeamCode/libs/desktop/libopencv_java490" + libSuffix + ".dylib";
        System.load(libPath);
    }

    public static void main(String[] args) throws Exception {
        String baseDir = System.getProperty("user.dir") + "/data/FIRST/";
        String framesDir = baseDir + "frames_1772255220/";
        String imuFile = baseDir + "imu_1772255220.txt";

        // 1. Parse IMU Data
        TreeMap<Long, Double> imuData = new TreeMap<>();
        double initialRawYaw = Double.NaN;
        try (BufferedReader br = new BufferedReader(new FileReader(imuFile))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 2) continue;
                long ts = (long) Double.parseDouble(parts[0]);
                double yaw = Double.parseDouble(parts[1]);
                if (Double.isNaN(initialRawYaw)) initialRawYaw = yaw;
                imuData.put(ts, yaw);
            }
        }
        
        // Initial yaw logic: start should be 0 or 180.
        // If 80.44, and we want it to be 180, offset is 80.44 - 180.
        // Let's assume start is 180 for this run based on previous context.
        double targetStartYaw = 180.0;
        double yawOffset = initialRawYaw - targetStartYaw;
        System.out.println("Initial Raw Yaw: " + initialRawYaw + ", Target: " + targetStartYaw + ", Offset: " + yawOffset);

        // 2. Setup Localizer
        Map<Integer, VisionLocalizer.VisionPose3D> fieldMap = VisionLocalizer.DECODE_FIELD_MAP;
        Mat cameraMatrix = VisionLocalizer.getArducamMatrix();
        MatOfDouble distCoeffs = VisionLocalizer.getArducamDistCoeffs();
        VisionLocalizer localizer = new VisionLocalizer(fieldMap, cameraMatrix, distCoeffs, VisionLocalizer.ARDUCAM_TRANSFORM);

        Dictionary dictionary = Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11);
        ArucoDetector detector = new ArucoDetector(dictionary);

        // 3. Get Sorted Image Files
        File dir = new File(framesDir);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".jpg"));
        Arrays.sort(files, Comparator.comparing(File::getName));

        // 4. Visualization Setup (Zoomed to 3 fields wide)
        Mat fieldDisplay = new Mat(800, 800, CvType.CV_8UC3);
        List<Point> robotPath = new ArrayList<>();
        double viewAreaInches = 144.0 * 3.0; // 432 inches
        double scale = 800.0 / viewAreaInches;
        Point center = new Point(400, 400);

        Mat frame = new Mat();
        Mat resized = new Mat();

        System.out.println("Processing " + files.length + " frames. SPACE to pause.");

        boolean isPaused = false;
        for (int f = 0; f < files.length; f++) {
            File file = files[f];
            String name = file.getName();
            // Extract timestamp from frame_seq_ts.jpg
            String[] nameParts = name.replace(".jpg", "").split("_");
            long frameTs = Long.parseLong(nameParts[nameParts.length - 1]);

            // Find nearest IMU
            Map.Entry<Long, Double> entry = imuData.floorEntry(frameTs);
            if (entry == null) entry = imuData.ceilingEntry(frameTs);
            double rawYaw = (entry != null) ? entry.getValue() : initialRawYaw;
            double correctedYaw = rawYaw - yawOffset;

            frame = Imgcodecs.imread(file.getAbsolutePath());
            if (frame.empty()) continue;
            localizer.prepareFrame(frame, resized);

            // Detect
            Mat ids = new Mat();
            List<Mat> corners = new ArrayList<>();
            detector.detectMarkers(resized, corners, ids);

            List<VisionLocalizer.Detection> detections = new ArrayList<>();
            if (!ids.empty()) {
                for (int i = 0; i < ids.rows(); i++) {
                    Mat cornerMat = corners.get(i);
                    Point[] pts = new Point[4];
                    for (int j = 0; j < 4; j++) pts[j] = new Point(cornerMat.get(0, j)[0], cornerMat.get(0, j)[1]);
                    detections.add(new VisionLocalizer.Detection((int)ids.get(i,0)[0], pts));
                }
            }

            // Estimate Pose WITH IMU
            VisionLocalizer.VisionPose3D robotPose = localizer.estimateRobotPose(detections, Math.toRadians(correctedYaw));

            // --- Render ---
            fieldDisplay.setTo(new Scalar(20, 20, 20));
            double fHalf = 72.0 * scale;
            Imgproc.rectangle(fieldDisplay, new Point(center.x - fHalf, center.y - fHalf), new Point(center.x + fHalf, center.y + fHalf), new Scalar(100, 100, 100), 2);
            
            if (robotPose != null) {
                Point p = new Point(center.x + robotPose.x * scale, center.y - robotPose.y * scale);
                robotPath.add(p);
            }

            // Draw Path
            for (int i = 1; i < robotPath.size(); i++) Imgproc.line(fieldDisplay, robotPath.get(i-1), robotPath.get(i), new Scalar(0, 255, 0), 1);

            // Tags
            for (Map.Entry<Integer, VisionLocalizer.VisionPose3D> tagEntry : fieldMap.entrySet()) {
                Point p = new Point(center.x + tagEntry.getValue().x * scale, center.y - tagEntry.getValue().y * scale);
                Imgproc.circle(fieldDisplay, p, 4, new Scalar(0, 0, 255), -1);
            }

            String info = String.format("Frame: %d, Yaw: %.1f", f, correctedYaw);
            Imgproc.putText(resized, info, new Point(30, 50), 0, 0.8, new Scalar(0, 255, 0), 2);
            if (robotPose != null) {
                String poseStr = String.format("X: %.1f, Y: %.1f", robotPose.x, robotPose.y);
                Imgproc.putText(resized, poseStr, new Point(30, 90), 0, 0.8, new Scalar(255, 255, 0), 2);
            }

            HighGui.imshow("Image Sequence", resized);
            HighGui.imshow("Field Path (IMU Corrected)", fieldDisplay);

            int key = HighGui.waitKey(isPaused ? 0 : 200);
            if (key == 113 || key == 27) break;
            if (key == 32) isPaused = !isPaused;
        }

        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
