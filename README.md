# Vision Transformer (ViT) vs. CNN on CIFAR-10

This project implements and compares a standard Convolutional Neural Network (CNN) and a Vision Transformer (ViT) for image classification on the CIFAR-10 dataset. The primary objective is to evaluate the performance of ViT in comparison to traditional CNNs for small-scale image classification tasks. For this project, the comparison is focused on a subset of CIFAR-10: **cats vs. deer**.

## Table of Contents

1.  [Objectives](#objectives)
2.  [Dataset](#dataset)
3.  [Project Structure](#project-structure-example)
4.  [Setup](#setup)
5.  [Implementation Details](#implementation-details)
    *   [CNN Model](#cnn-model)
    *   [Vision Transformer (ViT) Model](#vision-transformer-vit-model)
    *   [Pre-trained ViT (Optional)](#pre-trained-vit-optional)
6.  [Training](#training)
7.  [Evaluation](#evaluation)
8.  [Comparison and Analysis](#comparison-and-analysis)
9.  [Results (Expected/Observed)](#results-expectedobserved)
10. [Running the Notebook](#running-the-notebook)
11. [Future Work](#future-work)
12. [Acknowledgements](#acknowledgements)

## Objectives

*   Implement a CNN model suitable for CIFAR-10 (cats vs. deer) classification, incorporating best practices.
*   Implement a Vision Transformer (ViT) model based on the original ViT paper, adapted for CIFAR-10 (cats vs. deer).
*   Optionally, fine-tune a lightweight pre-trained ViT model on the dataset.
*   Evaluate all models on metrics including accuracy, precision, recall, F1-score, training/validation loss curves, and confusion matrices.
*   Compare the models in terms of:
    *   Classification performance and generalization.
    *   Training time and complexity.
    *   Model size and number of parameters.
    *   Effectiveness of attention-based modeling (ViT) versus spatial modeling (CNNs).

## Dataset

The project uses the **CIFAR-10 dataset**.
*   **Full Dataset URL:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
*   For this implementation, the dataset is **filtered to a subset of 2 classes: cats (class 3) and deer (class 4)** to speed up experimentation. This results in:
    *   10,000 training samples (5,000 per class)
    *   2,000 testing samples (1,000 per class)
*   Images are 32x32 color images.


## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/[your-username]/[your-repo-name].git
    cd vit-vs-cnn-cifar10
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The notebook `genai-a2-q4.ipynb` uses TensorFlow and other common libraries.
    ```bash
    pip install -r requirements.txt
    ```
    Create a `requirements.txt` file with the following (or based on your exact imports):
    ```
    tensorflow
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    # Add Pillow, opencv-python if used directly for image processing
    # Add datasets, transformers, timm if using pre-trained models
    ```

4.  **Dataset:**
    The CIFAR-10 dataset is automatically downloaded by `tensorflow.keras.datasets.cifar10.load_data()` within the notebook.

## Implementation Details

### CNN Model

*   **Architecture:** A custom CNN designed for 32x32 images.
    *   Three convolutional blocks with increasing filters (32, 64, 128).
    *   Each block contains:
        *   Two convolutional layers (3x3 kernel, 'same' padding, ReLU activation).
        *   Batch Normalization after each convolutional layer.
        *   Max Pooling (2x2) for downsampling.
        *   Dropout for regularization (rates: 0.2, 0.3, 0.4 progressively).
    *   Flatten layer.
    *   Dense layer (128 units, ReLU, L2 kernel regularizer).
    *   Batch Normalization.
    *   Dropout (0.5).
    *   Output Dense layer (1 unit, sigmoid activation for binary classification).
*   **Optimizer:** Adam (learning rate 0.001).
*   **Loss Function:** Binary Crossentropy.
*   **Data Preprocessing:** Rescaling (1./255), and `tf.data.Dataset` augmentations (random flip, random brightness).

### Vision Transformer (ViT) Model

*   **Architecture:** A custom ViT based on the original paper, adapted for 32x32 images.
    *   **Input Shape:** (32, 32, 3).
    *   **Patch Extraction:**
        *   `PatchExtractor` layer.
        *   Patch Size: 4x4 (resulting in (32/4)^2 = 64 patches per image).
    *   **Patch Encoding:**
        *   `PatchEncoder` layer.
        *   Linear projection of flattened patches (4x4x3=48 dimensions) to `projection_dim` (e.g., 64).
        *   Learnable positional embeddings added to patch embeddings.
    *   **Transformer Encoder:**
        *   Number of Layers: 6 (configurable).
        *   Each layer consists of:
            *   Layer Normalization.
            *   Multi-Head Self-Attention (MHSA) (e.g., 4 heads, `key_dim` = `projection_dim`).
            *   Skip Connection.
            *   Layer Normalization.
            *   MLP block (Dense layer with GELU activation).
            *   Skip Connection.
    *   **Classification Head:**
        *   Layer Normalization.
        *   Flatten.
        *   MLP (Dense layer, e.g., 128 units, GELU, Dropout).
        *   Output Dense layer (1 unit, sigmoid activation).
*   **Optimizer:** AdamW (learning rate 3e-4, weight decay 0.0001).
*   **Loss Function:** Binary Crossentropy.
*   **Data Preprocessing (ViT-specific):**
    *   Images normalized to [0,1] then standardized using ImageNet statistics (mean/std).
    *   Augmentations: `RandomRotation`, `RandomZoom`, `RandomContrast`.

### Pre-trained ViT (Optional)

*   The project allows for the optional use of a lightweight pre-trained ViT model (e.g., from Hugging Face `transformers` or `timm`).
*   This would involve loading the pre-trained model, replacing its classification head, and fine-tuning it on the CIFAR-10 cats vs. deer subset.
*   This part is not explicitly detailed in the provided notebook snippets but is an objective.

## Training

*   Training is performed within the `genai-a2-q4.ipynb` notebook.
*   **Key Training Practices:**
    *   **Data Augmentation:** Applied to increase dataset robustness (details specific to CNN and ViT).
    *   **Callbacks:**
        *   `EarlyStopping`: Monitors validation loss/accuracy to prevent overfitting and restore best weights.
        *   `ReduceLROnPlateau`: Reduces learning rate if validation performance stagnates.
        *   `ModelCheckpoint`: Saves the best model weights during training (e.g., `best_cnn.keras`, `best_vit.keras`).
    *   **Hyperparameter Tuning:** The notebook experiments with hyperparameters like learning rate, batch size, optimizer (Adam for CNN, AdamW for ViT), dropout rates, and ViT-specific parameters (projection dimension, number of layers/heads).
    *   **Epochs:** CNN trained for ~100 epochs (with early stopping), ViT trained for ~200 epochs (with early stopping). Batch size typically 64.

## Evaluation

Models are evaluated on the test set using the following metrics:
*   **Accuracy**
*   **Precision, Recall, F1-Score** (per class and weighted average)
*   **Training and Validation Loss Curves**
*   **Confusion Matrix**

The notebook includes functions to:
*   Plot training history (accuracy and loss curves).
*   Generate and display confusion matrices using `seaborn`.
*   Print classification reports from `sklearn.metrics`.
*   Optionally visualize attention maps for the ViT (though the specific layer index for attention output needs to be correctly identified in `vit_model.layers[6].output`).

## Comparison and Analysis

The notebook provides a comparative analysis, including:
1.  **Performance Metrics Table:** Comparing Accuracy, Precision, Recall, and F1-Score for CNN and ViT.
2.  **Training Curves Visualization:** Side-by-side or separate plots of training/validation accuracy and loss for both models.
3.  **Discussion Points:**
    *   **Accuracy & Generalization:** Which model performs better and generalizes well to the test set.
    *   **Training Time & Complexity:** Subjective comparison of how long each model takes to train and the inherent complexity of implementation.
    *   **Model Size & Parameters:** CNN model has ~550K parameters, ViT has ~950K parameters.
    *   **Effectiveness of Modeling:** Discussion on why CNNs might be more effective on small datasets due to their inductive biases (locality, translation equivariance) versus ViTs which require more data to learn these patterns.

## Results (Expected/Observed)

Based on the analysis in the notebook:
*   **CNN Performance:** Achieved a test accuracy of approximately **93.5% - 93.6%** on the cats vs. deer task. It shows good convergence and generalization.
*   **ViT Performance:** Achieved a test accuracy of approximately **77.0%** on the cats vs. deer task. The ViT model, trained from scratch on this small subset, underperforms the CNN. This is expected as ViTs are generally data-hungry and benefit more from large-scale pre-training.
*   **Key Observations:**
    *   CNNs, with their built-in spatial inductive biases, are highly effective for small-scale image classification tasks like this subset of CIFAR-10.
    *   ViTs, when trained from scratch on limited data, struggle to match CNN performance because they lack these biases and need to learn spatial relationships from patches, which requires more data.
    *   The ViT training curves show more volatility, indicating a more challenging optimization landscape on small data.

## Running the Notebook

1.  Ensure all dependencies from `requirements.txt` are installed in your Python environment.
2.  Open and run the `genai-a2-q4.ipynb` notebook in a Jupyter environment (Jupyter Lab, Jupyter Notebook, Google Colab, Kaggle Notebooks).
3.  The notebook is structured to:
    *   Load and preprocess the CIFAR-10 dataset (filtered for cats and deer).
    *   Define, compile, and train the CNN model.
    *   Define, compile, and train the ViT model.
    *   Evaluate both models and display metrics, curves, and confusion matrices.
    *   Provide a comparative analysis.
4.  It is recommended to use a GPU-enabled environment for faster training, especially for the ViT model. The notebook is configured to use a GPU if available (e.g., on Kaggle or Colab).

## Future Work

*   Implement and compare a lightweight pre-trained ViT fine-tuned on this task.
*   Experiment with different patch sizes for ViT (e.g., 8x8, though 4x4 is likely optimal for 32x32).
*   More extensive hyperparameter tuning for both models.
*   Train on the full CIFAR-10 dataset (10 classes) to see if ViT performance improves with more data diversity.
*   Explore hybrid CNN-ViT architectures.
*   Investigate different positional embedding schemes for ViT.

## Acknowledgements

*   The original Vision Transformer paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.
*   CIFAR-10 Dataset: Krizhevsky, A.
*   TensorFlow and Keras libraries.
*   Scikit-learn, Matplotlib, Seaborn for evaluation and visualization.
