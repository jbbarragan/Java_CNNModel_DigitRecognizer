/*
 * @File: CNNModel.java
 * @author: Joshua Barragán
 * @date: 2025-06
 * @brief: Clase para definir y manejar el modelo CNN para reconocimiento de dígitos.
 * 
*/
package app;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

public class CNNModel {

    private MultiLayerNetwork model;
    private static final String MODEL_PATH = "trained_model.zip";

    public CNNModel() {
        try {
            File modelFile = new File(MODEL_PATH);
            if (modelFile.exists()) {
                System.out.println("🔁 Modelo cargado desde disco...");
                model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            } else {
                System.out.println("🧠 Entrenando nuevo modelo...");
                model = createModel();
            }
        } catch (IOException e) {
            throw new RuntimeException("Error cargando o creando el modelo", e);
        }
    }

    private MultiLayerNetwork createModel() {
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.006);
        lrSchedule.put(1, 0.004);
        lrSchedule.put(2, 0.0015);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123) 
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, lrSchedule))) 
                .activation(Activation.IDENTITY) 
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()

                .layer(new ConvolutionLayer.Builder(3, 3).nIn(1).nOut(64).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer(Activation.RELU))
                .layer(new ConvolutionLayer.Builder(3, 3).nOut(64).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer(Activation.RELU))
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build())

                .layer(new ConvolutionLayer.Builder(3, 3).nOut(64).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer(Activation.RELU))
                .layer(new ConvolutionLayer.Builder(3, 3).nOut(64).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer(Activation.RELU))
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build())

                .layer(new DenseLayer.Builder().nOut(1024).activation(Activation.RELU).build())
                .layer(new DropoutLayer(0.4)) 


                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nOut(10)
                        .build())

                .setInputType(InputType.convolutionalFlat(28, 28, 1)) // Imágenes 28x28, escala de grises
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        return model;
    }

    public void train(org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainData, int epochs) {
        for (int i = 0; i < epochs; i++) {
            System.out.println("🔁 Epoch " + (i + 1) + "/" + epochs);
            model.fit(trainData);
        }

        
        try {
            ModelSerializer.writeModel(model, new File(MODEL_PATH), true);
            System.out.println("✅ Modelo guardado en disco: " + MODEL_PATH);
        } catch (IOException e) {
            System.err.println("❌ Error guardando el modelo: " + e.getMessage());
        }
    }

    public int predict(INDArray input) {
        return model.predict(input)[0];
    }

    public MultiLayerNetwork getModel() {
        return model;
    }
}
