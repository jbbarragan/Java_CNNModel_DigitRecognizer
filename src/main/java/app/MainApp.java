/*
 * @File: MainApp.java
 * @author: Joshua Barragán
 * @date: 2025-06
 * @brief: Clase principal que inicia la aplicación de reconocimiento de dígitos.
 * 
*/
package app;

import java.awt.image.BufferedImage;
import java.io.File;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class MainApp {

    private static final String MODEL_PATH = "trained_model.zip";

    public static void main(String[] args) throws Exception {
        
        System.loadLibrary("opencv_java4110");

        CameraHandler camera = new CameraHandler();
        CNNModel cnn = new CNNModel();

        
        File modelFile = new File(MODEL_PATH);
        if (!modelFile.exists()) {
            System.out.println("⚙️ Entrenando modelo desde MainApp...");
            Trainer.trainModel(cnn);
        } else {
            System.out.println("✅ Modelo ya entrenado, usando directamente.");
        }

        DisplayWindow window = new DisplayWindow();

        if (!camera.isOpened()) {
            System.out.println("No se pudo abrir la cámara.");
            return;
        }

        while (true) {
            Mat frame = camera.getFrame();

            if (frame == null || frame.empty()) {
                System.err.println("Error: Frame de cámara nulo o vacío.");
                continue;
            }

            INDArray input = DigitPreprocessor.preprocess(frame);
            int predictedDigit = cnn.predict(input);

            BufferedImage image = matToBufferedImage(frame);
            if (image != null) {
                window.updateImage(image, predictedDigit);
            }

            Thread.sleep(500);  
        }
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            System.err.println("Error: El objeto Mat está vacío o es nulo.");
            return null;
        }

        try {
            if (mat.type() != CvType.CV_8UC3) {
                System.err.println("Error: Mat no es del tipo CV_8UC3 (8-bit, 3 canales). Tipo actual: " + mat.type());
                return null;
            }

            int width = mat.width();
            int height = mat.height();
            int channels = mat.channels();

            byte[] sourcePixels = new byte[width * height * channels];
            mat.get(0, 0, sourcePixels);

            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
            image.getRaster().setDataElements(0, 0, width, height, sourcePixels);

            return image;
        } catch (Exception e) {
            System.err.println("Error al convertir Mat a BufferedImage: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}