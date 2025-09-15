# Multi-Layer Perceptron (MLP) on MNIST

## Project Overview
This project implements a **Multi-Layer Perceptron (MLP)** using **TensorFlow/Keras** to classify handwritten digits from the **MNIST dataset**.

The MLP has two hidden layers with **sigmoid activations** and a **softmax output layer** for 10-class classification.

## Dataset
- MNIST dataset: 28x28 grayscale images of handwritten digits (0–9)
- Loaded directly from `tensorflow.keras.datasets.mnist`
- Normalized to the range [0, 1]

## Model Architecture
| Layer       | Neurons | Activation |
|------------|---------|-----------|
| Input      | 28x28  | -         |
| Flatten    | 784    | -         |
| Dense 1    | 256    | sigmoid   |
| Dense 2    | 128    | sigmoid   |
| Output     | 10     | softmax   |

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install tensorflow numpy matplotlib
