# Java KNN Iris Classification

Iris flower classification using the K-Nearest Neighbors (KNN) algorithm, implemented in Java.

**Project 2 - DecodeLabs Internship**

## About the Project

This project classifies iris flowers into three species (Setosa, Versicolor, Virginica) using four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Features

- Loading the Iris dataset (150 samples) from a CSV file
- Feature scaling (standardization), learned from training data only
- 80/20 train-test split (120 training / 30 testing samples)
- K-Nearest Neighbors with Euclidean distance
- 5-fold cross-validation to select the best K (K = 1, 3, 5, 7, 9)
- Confusion matrix
- Precision, Recall and F1-score for each class
- Prediction for a new flower

## How It Works

1. Load the Iris dataset
2. Shuffle the data and split it 80% training / 20% testing
3. Standardize the features using the training data
4. Compare different K values with 5-fold cross-validation
5. Test the model with the best K
6. Calculate the confusion matrix, precision, recall and F1-score
7. Predict the class of a new flower (5.1, 3.5, 1.4, 0.2)

## Project Structure

```
java-knn-iris-classification
├── data
│   └── iris.csv
├── src
│   └── Project2_KNN_Iris.java
└── README.md
```

## How to Run

1. Clone the repository:
```
   git clone https://github.com/Tayaba-bibi/java-knn-iris-classification.git
```
2. Open the project in IntelliJ IDEA (or any Java IDE)
3. Make sure the `data/iris.csv` file is in the project root folder
4. Run `src/Project2_KNN_Iris.java`

## Results

### Dataset Split

| Set | Samples |
|---|---|
| Training | 120 (80%) |
| Testing | 30 (20%) |

### K Value Comparison (5-Fold Cross Validation)

| K | Accuracy |
|---|---|
| 1 | 95.00% |
| 3 | 95.83% |
| 5 | 95.83% |
| 7 | 95.83% |
| 9 | 95.83% |

**Selected K = 3** (best cross-validation accuracy: 95.83%)

### Final Test Result

**Test Accuracy = 93.33%** (28 out of 30 test samples classified correctly)

### Confusion Matrix

| Actual \ Predicted | Setosa | Versicolor | Virginica |
|---|---|---|---|
| Setosa | 7 | 0 | 0 |
| Versicolor | 0 | 9 | 1 |
| Virginica | 0 | 1 | 12 |

### Evaluation Metrics

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 0.90 | 0.90 | 0.90 |
| Virginica | 0.92 | 0.92 | 0.92 |

### Sample Prediction

| Feature | Value |
|---|---|
| Sepal Length | 5.1 |
| Sepal Width | 3.5 |
| Petal Length | 1.4 |
| Petal Width | 0.2 |

**Predicted class: Setosa**

## What I Learned

- How the K-Nearest Neighbors algorithm works
- Why feature scaling matters for distance-based models
- How to avoid data leakage by fitting the scaler on training data only
- How to choose K using cross-validation
- How to evaluate a classifier with a confusion matrix, precision, recall and F1-score

## Future Improvements

- Fit the scaler inside each cross-validation fold
- Add more distance metrics
- Add a simple user input option for new flowers

## Author

**Tayaba bibi**
BS Computer Science Student
