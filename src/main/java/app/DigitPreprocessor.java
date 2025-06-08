/*
 * @File: DigitPreprocessor.java
 * @author: Joshua Barragán
 * @date: 2025-06
 * @brief: Clase para preprocesar imágenes de dígitos antes de la inferencia.
 * 
*/
package app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class DigitPreprocessor {

    public static INDArray preprocess(Mat frame) {
        Mat gray = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);

        Mat binary = new Mat();
        Imgproc.adaptiveThreshold(gray, binary, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  Imgproc.THRESH_BINARY_INV, 11, 2);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binary.clone(), contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat digitMat = Mat.zeros(new Size(28, 28), CvType.CV_8UC1);

        if (!contours.isEmpty()) {
            MatOfPoint biggestContour = Collections.max(contours, Comparator.comparingDouble(Imgproc::contourArea));
            Rect bbox = Imgproc.boundingRect(biggestContour);

            Mat roi = new Mat(binary, bbox);

            Mat scaled = new Mat();
            Imgproc.resize(roi, scaled, new Size(20, 20));

            int x = (28 - scaled.cols()) / 2;
            int y = (28 - scaled.rows()) / 2;
            scaled.copyTo(digitMat.submat(y, y + scaled.rows(), x, x + scaled.cols()));
        }

        byte[] data = new byte[(int) (digitMat.total() * digitMat.channels())];
        digitMat.get(0, 0, data);

        float[] floatData = new float[data.length];
        float mean = 0.1307f;
        float std = 0.3081f;

        for (int i = 0; i < data.length; i++) {
            float normalized = (data[i] & 0xFF) / 255.0f;
            floatData[i] = (normalized - mean) / std;
        }

        return Nd4j.create(floatData).reshape(1, 1, 28, 28);
    }
}
