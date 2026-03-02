package org.firstinspires.ftc.teamcode.vision;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class VisionLocalizerVideoTest {

    static {
        String arch = System.getProperty("os.arch").toLowerCase();
        String libSuffix = arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64";
        String libPath = System.getProperty("user.dir") + "/TeamCode/libs/desktop/libopencv_java490" + (arch.contains("aarch64") || arch.contains("arm64") ? "_arm64" : "_x64") + (System.getProperty("os.name").toLowerCase().contains("mac") ? ".dylib" : ".so");
        System.load(libPath);
    }

    public static void main(String[] args) {
        String videoPath = System.getProperty("user.dir") + "/data/vision_localizer/arducam1.mp4";
        String outputDir = System.getProperty("user.dir") + "/data/vision_localizer/";

        Map<Integer, VisionLocalizer.VisionPose3D> fieldMap = VisionLocalizer.DECODE_FIELD_MAP;
        Mat cameraMatrix = VisionLocalizer.getArducamMatrix();
        MatOfDouble distCoeffs = VisionLocalizer.getArducamDistCoeffs();
        VisionLocalizer localizer = new VisionLocalizer(fieldMap, cameraMatrix, distCoeffs, VisionLocalizer.ARDUCAM_TRANSFORM);

        Dictionary dictionary = Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11);
        ArucoDetector detector = new ArucoDetector(dictionary);

        VideoCapture cap = new VideoCapture(videoPath);
        Mat frame = new Mat(), resized = new Mat();
        
        // Visualization Setup (5 fields wide)
        Mat fieldDisplay = new Mat(800, 800, CvType.CV_8UC3);
        List<Point> path20 = new ArrayList<>(), path24 = new ArrayList<>();
        double scale = 800.0 / (144.0 * 5.0);
        Point center = new Point(400, 400);

        final boolean[] shouldQuit = {false};
        Thread terminalListener = new Thread(() -> {
            try { System.in.read(); shouldQuit[0] = true; } catch (Exception e) {}
        });
        terminalListener.setDaemon(true);
        terminalListener.start();

        int frameCount = 0;
        boolean isPaused = false;
        int lastDetections = 0;

        while (!shouldQuit[0]) {
            if (!isPaused) {
                if (cap.read(frame)) {
                    if (frame.empty()) break;
                    frameCount++;
                    localizer.prepareFrame(frame, resized);

                    Mat ids = new Mat();
                    List<Mat> corners = new ArrayList<>();
                    detector.detectMarkers(resized, corners, ids);
                    lastDetections = ids.empty() ? 0 : ids.rows();

                    // --- Render Map ---
                    fieldDisplay.setTo(new Scalar(20, 20, 20));
                    double fHalf = 72.0 * scale;
                    Imgproc.rectangle(fieldDisplay, new Point(center.x - fHalf, center.y - fHalf), new Point(center.x + fHalf, center.y + fHalf), new Scalar(80, 80, 80), 2);
                    
                    Imgproc.putText(fieldDisplay, "BACK WALL (-X)", new Point(center.x - fHalf - 120, center.y), 0, 0.5, new Scalar(100, 100, 255), 1);
                    Imgproc.putText(fieldDisplay, "AUDIENCE (+X)", new Point(center.x + fHalf + 10, center.y), 0, 0.5, new Scalar(100, 100, 255), 1);
                    Imgproc.putText(fieldDisplay, "RED ALLIANCE (+Y)", new Point(center.x - 50, center.y - fHalf - 10), 0, 0.5, new Scalar(0, 0, 255), 1);
                    Imgproc.putText(fieldDisplay, "BLUE ALLIANCE (-Y)", new Point(center.x - 50, center.y + fHalf + 20), 0, 0.5, new Scalar(255, 0, 0), 1);

                    for (Map.Entry<Integer, VisionLocalizer.VisionPose3D> entry : fieldMap.entrySet()) {
                        int id = entry.getKey();
                        VisionLocalizer.VisionPose3D t = entry.getValue();
                        Point p = new Point(center.x + t.x * scale, center.y - t.y * scale);
                        Scalar tagColor = (id == 24) ? new Scalar(0, 0, 255) : new Scalar(255, 0, 0);
                        Imgproc.rectangle(fieldDisplay, new Point(p.x-5, p.y-5), new Point(p.x+5, p.y+5), tagColor, -1);
                        Point facing = new Point(p.x + Math.cos(t.yaw) * 15.0, p.y - Math.sin(t.yaw) * 15.0);
                        Imgproc.line(fieldDisplay, p, facing, new Scalar(255, 255, 255), 2);
                        Imgproc.putText(fieldDisplay, "" + id, new Point(p.x + 8, p.y + 5), 0, 0.4, new Scalar(255, 255, 255), 1);
                    }

                    if (lastDetections > 0) {
                        for (int i = 0; i < ids.rows(); i++) {
                            int id = (int)ids.get(i,0)[0];
                            Mat cornerMat = corners.get(i);
                            Point[] pts = new Point[4];
                            for (int j = 0; j < 4; j++) pts[j] = new Point(cornerMat.get(0, j)[0], cornerMat.get(0, j)[1]);
                            
                            List<VisionLocalizer.Detection> single = new ArrayList<>();
                            single.add(new VisionLocalizer.Detection(id, pts));
                            VisionLocalizer.VisionPose3D pose = localizer.estimateRobotPose(single, Math.toRadians(180.0));
                            if (pose != null) {
                                Point pix = new Point(center.x + pose.x * scale, center.y - pose.y * scale);
                                if (id == 20) path20.add(pix);
                                if (id == 24) path24.add(pix);
                            }
                        }
                    }

                    for (int i = 1; i < path20.size(); i++) Imgproc.line(fieldDisplay, path20.get(i-1), path20.get(i), new Scalar(255, 0, 0), 1);
                    for (int i = 1; i < path24.size(); i++) Imgproc.line(fieldDisplay, path24.get(i-1), path24.get(i), new Scalar(0, 0, 255), 1);
                    if (!path20.isEmpty()) Imgproc.circle(fieldDisplay, path20.get(path20.size()-1), 5, new Scalar(255, 0, 0), -1);
                    if (!path24.isEmpty()) Imgproc.circle(fieldDisplay, path24.get(path24.size()-1), 5, new Scalar(0, 0, 255), -1);

                    Imgproc.putText(resized, "Tags: " + lastDetections, new Point(30, 50), 0, 1.0, new Scalar(0, 255, 0), 2);
                } else break;
            }

            HighGui.imshow("Video", resized);
            HighGui.imshow("Field Map", fieldDisplay);
            int key = HighGui.waitKey((lastDetections > 0 || isPaused) ? 100 : 2);
            if (key == 113 || key == 27) break;
            if (key == 32 || key == 112) isPaused = !isPaused;
            if (key == 99) {
                String path = outputDir + "arducam1_" + frameCount + ".jpg";
                Imgcodecs.imwrite(path, resized);
                System.out.println("Captured: " + path);
            }
        }
        cap.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
