# Computational Intelligence - Academic Year 2024-2025

[![Course](https://img.shields.io/badge/Course-Computational_Intelligence_(MYE035)-blue.svg)]()
[![Institution](https://img.shields.io/badge/Institution-University_of_Ioannina-red.svg)]()
[![Language](https://img.shields.io/badge/Language-Java-orange.svg)]()

This repository contains the laboratory assignments and project reports for the Computational Intelligence course at the University of Ioannina, Department of Computer Science & Engineering. 

The algorithms for neural network classification and data clustering were implemented entirely from scratch in **Java**, strictly without the use of high-level machine learning libraries like Python's scikit-learn or TensorFlow.

**Team Members:**
* Giannis Fillis, AM: 5380
* Konstantinos Zois, AM: 5226

---

## 📖 Table of Contents
1. [Project 1: Multilayer Perceptron (MLP) Classification](#project-1-multilayer-perceptron-mlp-classification)
2. [Project 2: K-Means Clustering](#project-2-k-means-clustering)
3. [Compilation and Execution](#compilation-and-execution)
4. [Project Structure](#project-structure)

---

## 🧠 Project 1: Multilayer Perceptron (MLP) Classification

### Overview
The first exercise involves building from scratch two Multilayer Perceptron (MLP) models to solve a complex non-linear 4-class classification problem. 
* **PT2 Model:** MLP architecture with 2 hidden layers.
* **PT3 Model:** MLP architecture with 3 hidden layers.

### Dataset
* The synthetic dataset consists of 8,000 random 2D points (X1, X2) generated within a [-1, 1] x [-1, 1] square.
* The dataset is split equally into 4,000 points for the training set and 4,000 points for the test set.
* Points are classified into four distinct categories (C1, C2, C3, C4) based on a set of 10 complex geometric constraints (e.g., circular regions and specific Cartesian quadrants).
* Output categories are mapped using 1-out-of-K (one-hot) encoding.

### Implementation Details
* **Activation Functions:** Hyperbolic tangent (tanh) and ReLU functions are used in the hidden layers.
* **Training Algorithm:** Gradient descent with backpropagation.
* **Optimization:** Weight updating is performed using mini-batches. Experiments were conducted with batch sizes of B = N/20 and B = N/200.
* **Initialization:** Weights and biases are initialized randomly in the range (-1, 1). 

### Key Findings
* **PT2 Best Model:** Achieved 84.225% generalization ability (accuracy) using H1 = 20, H2 = 40 neurons, hyperbolic tangent activation, and batch size B = 20.
* **PT3 Best Model:** Achieved 80.025% generalization ability using H1 = 20, H2 = 40, H3 = 8 neurons, ReLU activation in the third layer, and batch size B = 200.
* **Conclusion:** The 2-hidden-layer model (PT2) was sufficiently capable for this dataset and performed slightly better than the 3-hidden-layer model (PT3) at lower neuron counts. 

---

## 📊 Project 2: K-Means Clustering

### Overview
The second exercise focuses on unsupervised learning, specifically implementing the k-means clustering algorithm to group spatial data.

### Dataset
* The dataset consists of 1,000 generated 2D points distributed across various rectangular coordinate boundaries.
* Although the ground-truth number of clusters is 8, the algorithm treats this as an unknown variable to evaluate clustering metrics.

### Implementation Details
* The algorithm randomly initializes M centroids by selecting actual points from the dataset.
* Point assignment is calculated using Euclidean distance mapping to the nearest centroid.
* Centroids are recalculated by computing the mean coordinates of all assigned points.
* The algorithm loops until convergence (when no centroid changes position).
* For each given M (number of clusters), the program executes 20 independent runs and retains the configuration with the minimum clustering error.

### Key Findings
* Evaluated models for M = 4, 6, 8, 10, and 12 clusters.
* **Clustering Error Results:** * M = 4: Error ≈ 188.24
  * M = 6: Error ≈ 109.71
  * M = 8: Error ≈ 39.56
  * M = 10: Error ≈ 34.75
  * M = 12: Error ≈ 31.07
* **Conclusion:** Clustering error continuously decreases as M increases, making it a flawed standalone metric for identifying the true number of clusters (which is 8).

---

## 🚀 Compilation and Execution

Since the code is written purely in Java without external dependencies, you can compile and run the source files directly via the terminal.

### Project 1 (PT2 & PT3)
Navigate to the Project 1 directory and use the `javac` compiler:
```bash
# Compile the PT2 model
javac PT2.java
# Run the PT2 model
java PT2
```
*(The exact same process applies to the `PT3.java` file. Parameters are configured directly within the code prior to compilation).*

### Project 2 (K-Means)
Navigate to the Project 2 directory:
```bash
# Compile the K-Means algorithm
javac PO.java
# Run the algorithm
java PO
```
