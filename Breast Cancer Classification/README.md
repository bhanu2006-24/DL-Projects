# 🎗️ Breast Cancer Classification using Deep Learning

**Type**: Binary Classification  
**Model**: Artificial Neural Network (ANN)  
**Framework**: TensorFlow/Keras  
**Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset

**[Open in Colab](https://colab.research.google.com/drive/10wjsXNVWc-xq8qs_iCRVX6Hic9eLekND?usp=sharing)**

## 📝 Description

This project implements a Deep Learning model to automatically classify breast mass samples as either **Malignant** (suspicious) or **Benign** (non-cancerous). Early diagnosis of breast cancer can significantly improve survival rates, and this project demonstrates how neural networks can assist medical professionals in making accurate diagnoses based on quantitative features computed from digitized images of a fine needle aspirate (FNA) of a breast mass.

## 📂 Dataset Details

The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Data Set**, available directly through `sklearn.datasets`.

- **Samples**: 569 instances
- **Features**: 30 numeric, predictive attributes (e.g., radius, texture, perimeter, area, smoothness, compactness, etc.)
- **Classes**:
  - `0`: Malignant
  - `1`: Benign

## 🧠 Model Architecture

The model is a Feed-Forward Neural Network (ANN) built using Keras:

1.  **Input Layer**: Flattens the input features (30 standard features).
2.  **Hidden Layer**: Dense layer with **20 neurons** using **ReLU** (Rectified Linear Unit) activation function to introduce non-linearity.
3.  **Output Layer**: Dense layer with **2 neurons** using **Sigmoid** or **Softmax** logic for binary classification probability.

**Compilation**:

- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

## 📊 Performance & Results

The model was trained for 10 epochs with a validation split.

- **Test Accuracy**: **~95.6%**
- **Test Loss**: **~0.158**

The loss and accuracy curves demonstrate that the model converges well without significant overfitting, achieving reliable performance on unseen test data.

## 🗝️ Key Concepts Learned

- **Data Preprocessing**:
  - Loading data from Scikit-Learn.
  - Analyzing class distribution (Benign vs Malignant).
  - **Standardization**: Using `StandardScaler` to scale features to a mean of 0 and standard deviation of 1, which is crucial for Neural Network convergence.
- **Neural Network Implementation**:
  - Defining a Sequential model in Keras.
  - Understanding layer connectivity and activation functions.
- **Training & Evaluation**:
  - Monitoring training history (loss/accuracy over epochs).
  - Evaluating the model on a held-out test set.
  - Visualizing performance metrics using Matplotlib.

## 🛠️ Technologies Used

- **Python**: Core programming language.
- **TensorFlow & Keras**: For building and training the deep learning model.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Matplotlib**: For plotting accuracy and loss graphs.
- **Scikit-Learn**: For dataset loading, train-test splitting, and preprocessing.

## 🚀 How to Run

1.  Clone the repository and navigate to the folder.
2.  Ensure you have the required libraries installed:
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn
    ```
3.  Run the Jupyter Notebook `Breast_Cancer_Classification.ipynb` or open it directly in [Google Colab](https://colab.research.google.com/drive/10wjsXNVWc-xq8qs_iCRVX6Hic9eLekND?usp=sharing).

---

_Created as part of my Deep Learning projects portfolio._
