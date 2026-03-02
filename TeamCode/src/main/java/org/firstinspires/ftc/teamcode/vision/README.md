# Vision Desktop Test Bench

This folder contains vision processing logic for the robot, along with a desktop-based testing environment that allows for rapid iteration on your laptop (macOS) without deploying to the Rev Control Hub.

## Directory Structure
- `VisionLocalizer.java`: MegaTag 2 algorithm (PnP + IMU) for robot relocalization.
- `VisionArtifactDetector.java`: Color segmentation and Homography projection for game element detection.
- `src/test/java/...`: Unit tests that run on your PC using standard Java.
- `libs/desktop/`: Pre-compiled OpenCV JAR and native libraries for macOS (x86_64 and ARM64).

## Getting Started (macOS)

### 1. Prerequisites
You need a Java Development Kit (JDK) installed. It's recommended to use the latest OpenJDK.

Install via Homebrew:
```bash
brew install openjdk
```

### 2. How to Run Tests
Open your terminal at the **project root** directory and run the following commands:

#### Compile the Vision and Test classes:
```bash
# Create build directory
mkdir -p build/vision_classes

# Compile Main Classes
/usr/local/opt/openjdk/bin/javac -cp "DecodeV3/TeamCode/libs/desktop/opencv-4.9.0-0.jar" 
    DecodeV3/TeamCode/src/main/java/org/firstinspires/ftc/teamcode/vision/VisionLocalizer.java 
    DecodeV3/TeamCode/src/main/java/org/firstinspires/ftc/teamcode/vision/VisionArtifactDetector.java 
    -d build/vision_classes

# Compile Test Classes
/usr/local/opt/openjdk/bin/javac -cp "DecodeV3/TeamCode/libs/desktop/opencv-4.9.0-0.jar:build/vision_classes" 
    DecodeV3/TeamCode/src/test/java/org/firstinspires/ftc/teamcode/vision/VisionLocalizerTest.java 
    DecodeV3/TeamCode/src/test/java/org/firstinspires/ftc/teamcode/vision/VisionArtifactDetectorTest.java 
    -d build/vision_classes
```

#### Run the Tests:
The tests will automatically detect whether you are on an Intel (x86_64) or Apple Silicon (ARM64) Mac and load the correct native library.

**Run Localizer Test:**
```bash
/usr/local/opt/openjdk/bin/java -cp "DecodeV3/TeamCode/libs/desktop/opencv-4.9.0-0.jar:build/vision_classes" 
    org.firstinspires.ftc.teamcode.vision.VisionLocalizerTest
```

**Run Artifact Detector Test:**
```bash
/usr/local/opt/openjdk/bin/java -cp "DecodeV3/TeamCode/libs/desktop/opencv-4.9.0-0.jar:build/vision_classes" 
    org.firstinspires.ftc.teamcode.vision.VisionArtifactDetectorTest
```

## Integration with Robot
The classes `VisionLocalizer` and `VisionArtifactDetector` are designed to be "Hardware Agnostic". 
To use them on the robot:
1. Call `estimateRobotPose()` or `detect()` from your `OpenCvPipeline` or `VisionProcessor`.
2. All OpenCV methods are fully compatible with the FTC Control Hub SDK.
