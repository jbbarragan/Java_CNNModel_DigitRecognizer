/*
 * @File: DisplayWindow.java
 * @author: Joshua Barragán
 * @date: 2025-06
 * @brief: Clase para crear una ventana que muestra la imagen capturada y la predicción del modelo.
 * 
*/
package app;

import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.image.BufferedImage;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingConstants;

public class DisplayWindow extends JFrame {
    private JLabel imageLabel;
    private JLabel predictionLabel;

    public DisplayWindow() {
        setTitle("Reconocedor de Dígitos");
        setSize(400, 400);
        setLayout(new BorderLayout());
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        imageLabel = new JLabel();
        predictionLabel = new JLabel("Predicción: ", SwingConstants.CENTER);
        predictionLabel.setFont(new Font("Arial", Font.BOLD, 24));

        add(imageLabel, BorderLayout.CENTER);
        add(predictionLabel, BorderLayout.SOUTH);

        setVisible(true);
    }

    public void updateImage(BufferedImage image, int prediction) {
        imageLabel.setIcon(new ImageIcon(image));
        predictionLabel.setText("Predicción: " + prediction);
    }
}
