/* 

 *este elemento es un extra si deseas interactuar con el modelo
 * @File: Interactive_main.java
 * @author: Joshua Barragán
 * @date: 2025-06
 * @brief: Clase principal que inicia la aplicacion y nos permite interactuar con el modelo para un "ajuste fino"
 * permite interaccion para mejorar predicciones del modelo
 * 
package app;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Scanner;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.util.ModelSerializer;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class Interactive_main {

    private static final String MODEL_PATH = "trained_model.zip";

    public static void main(String[] args) throws Exception {
        System.loadLibrary("opencv_java4110");

        CameraHandler camera = new CameraHandler();
        CNNModel cnn = new CNNModel();
        DisplayWindow window = new DisplayWindow();
        Scanner scanner = new Scanner(System.in);

        if (!camera.isOpened()) {
            System.out.println("No se pudo abrir la cámara.");
            return;
        }

        
        File modelFile = new File(MODEL_PATH);
        if (!modelFile.exists()) {
            System.out.println("⚙️ Entrenando modelo desde MainApp...");
            Trainer.trainModel(cnn);
        } else {
            System.out.println("✅ Modelo ya entrenado, usando directamente.");
        }

        System.out.println("Presiona '1' y luego ENTER para comenzar con las predicciones...");

        
        while (true) {
            String start = scanner.nextLine().trim();
            if (start.equals("1")) {
                System.out.println("Comenzando predicciones...");
                break;
            } else {
                System.out.println("Por favor, presiona '1' y ENTER para iniciar.");
            }
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

            System.out.print("¿Es correcta la predicción (" + predictedDigit + ")? [s/n]: ");
            String answer = scanner.nextLine().trim().toLowerCase();

            if (answer.equals("n")) {
                System.out.print("Ingrese el dígito correcto (0-9): ");
                String labelStr = scanner.nextLine().trim();

                try {
                    int label = Integer.parseInt(labelStr);
                    if (label < 0 || label > 9) {
                        System.out.println("Número inválido. Debe ser entre 0 y 9.");
                        continue;
                    }

                    INDArray labelArray = Nd4j.zeros(1, 10);
                    labelArray.putScalar(label, 1.0);
                    DataSet singleExample = new DataSet(input, labelArray);

                    cnn.getModel().fit(singleExample);
                    System.out.println("🔁 Modelo ajustado con corrección.");

                    try {
                        ModelSerializer.writeModel(cnn.getModel(), new File(MODEL_PATH), true);
                        System.out.println("💾 Modelo guardado automáticamente.");
                    } catch (Exception e) {
                        System.err.println("❌ Error al guardar el modelo: " + e.getMessage());
                    }

                } catch (NumberFormatException e) {
                    System.out.println("Entrada inválida. Intenta de nuevo.");
                }
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
*/