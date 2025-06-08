/*
 * @File: Trainer.java
 * @author: Joshua Barragán
 * @date: 2025-06
 * @brief: Clase para entrenar el modelo CNN con el dataset MNIST.
 * 
*/
package app;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class Trainer {

    public static void trainModel(CNNModel cnnModel) throws Exception {
        
        int batchSize = 64;
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);

        System.out.println("Entrenando modelo con MNIST...");
        cnnModel.train(mnistTrain, 3); 
        System.out.println("Entrenamiento terminado.");
    }
}

