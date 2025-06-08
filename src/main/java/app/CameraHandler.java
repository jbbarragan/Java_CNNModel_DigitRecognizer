/*
 * @File: CameraHandler.java
 * @author: Joshua Barragán
 * @date: 2025-06
 * @brief: Clase para manejar la cámara y procesar imágenes en tiempo real.
 * 
*/
package app;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class CameraHandler {
    private VideoCapture camera;

    public CameraHandler() {
        System.loadLibrary("opencv_java4110");
        camera = new VideoCapture(0); // cámara por defecto
    }

    public boolean isOpened() {
        return camera.isOpened();
    }

    public Mat getFrame() {
    Mat frame = new Mat();
    if (camera.isOpened()) {
        camera.read(frame);
    }

    if (frame.empty()) return frame;

    Mat gray = new Mat();
    Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

    Mat mask = new Mat();
    Core.inRange(gray, new Scalar(0), new Scalar(60), mask);   // negro
    Mat whiteMask = new Mat();
    Core.inRange(gray, new Scalar(200), new Scalar(255), whiteMask); // blanco

    Core.bitwise_or(mask, whiteMask, mask);

    Mat blurred = new Mat();
    Imgproc.GaussianBlur(frame, blurred, new Size(21, 21), 0);

    Mat maskInv = new Mat();
    Core.bitwise_not(mask, maskInv);

    Mat result = new Mat();
    frame.copyTo(result, mask);         
    blurred.copyTo(result, maskInv);    

    return result;
}


    public void release() {
        if (camera.isOpened()) {
            camera.release();
        }
    }
}
