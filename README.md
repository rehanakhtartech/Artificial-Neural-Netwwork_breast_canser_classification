# Breast Cancer Classification with an Artificial Neural Network (ANN)

A beginner-friendly machine learning project that trains an ANN to classify breast tumors as **malignant** or **benign**, then compares it with a Decision Tree and Naive Bayes and runs experiments to see how scaling, network size, and learning rate affect the results.

**Author:** Rehan Akhtar

---

## Project Overview

- **Task:** Binary classification (Malignant vs. Benign)
- **Dataset:** Breast Cancer Wisconsin dataset (569 patients, 30 numeric features)
- **Main model:** Multi-layer perceptron (ANN) built with scikit-learn's `MLPClassifier`
- **Compared against:** Decision Tree (CART) and Gaussian Naive Bayes
- **Positive class:** Malignant (the case we most want to catch, so **recall** is the key metric)

## Dataset

| Item | Value |
|---|---|
| Rows | 569 |
| Features | 30 (radius, texture, perimeter, area, smoothness, concavity, and more) |
| Target | `0` = Malignant (212), `1` = Benign (357) |
| Missing values | None |

The data comes from scikit-learn (`load_breast_cancer`) and is saved as `breast_cancer.csv`.

## Workflow

1. Load the CSV with pandas and explore it
2. Split into **70% train / 30% test** (stratified, `random_state=42`)
3. **Scale** the features with `StandardScaler` (fitted on the training data only, to avoid data leakage)
4. Build and train the ANN
5. Evaluate with accuracy, precision, recall, specificity, F1-score, and a confusion matrix
6. Compare with a Decision Tree and Naive Bayes on the same split
7. Run experiments on scaling, network size, and learning rate

## ANN Architecture

```
Input (30 features) -> Dense 16 (ReLU) -> Dense 8 (ReLU) -> Output (sigmoid)
```

| Setting | Value |
|---|---|
| Hidden layers | (16, 8) |
| Activation | ReLU |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Max epochs | 500 (stopped at 447) |

## Results

### ANN performance (test set, 171 patients)

| Metric | Score |
|---|---|
| Accuracy | 0.971 |
| Precision | 0.940 |
| Recall | 0.984 |
| F1-score | 0.962 |

**Confusion matrix** (rows = actual, columns = predicted):

| | Predicted Malignant | Predicted Benign |
|---|---|---|
| **Actual Malignant** | 63 | 1 |
| **Actual Benign** | 4 | 103 |

The ANN missed only **1 of 64** malignant cases.

### Model comparison

| Model | Accuracy | Precision | Recall | F1 | Missed malignant |
|---|---|---|---|---|---|
| Decision Tree (depth 4) | 0.924 | 0.905 | 0.891 | 0.898 | 7 |
| Naive Bayes | 0.947 | 0.966 | 0.891 | 0.927 | 7 |
| **ANN** | **0.971** | 0.940 | **0.984** | **0.962** | **1** |

### Experiments on the ANN

| Experiment | Test Acc | Recall | Missed |
|---|---|---|---|
| Baseline (16, 8) | 0.971 | 0.984 | 1 |
| **No feature scaling** | 0.924 | 0.859 | 9 |
| Tiny network (2) | 0.953 | 0.984 | 1 |
| Small network (4) | 0.959 | 0.969 | 2 |
| Big network (64, 32) | 0.977 | 0.984 | 1 |
| Huge network (128, 64, 32) | 0.971 | 0.984 | 1 |
| Low learning rate (0.0001) | 0.637 | 1.000 | 0 |
| High learning rate (0.1) | 0.971 | 0.984 | 1 |

## Key Learnings

- **Scaling is essential for ANNs.** Without it, recall dropped from 0.984 to 0.859.
- **Bigger is not always better.** Even a 2-neuron network reached 0.953 accuracy; larger networks mostly memorized the training data (train accuracy 1.000).
- **A learning rate that is too low stops the model from learning.** It also shows why one metric can mislead: recall was 1.000, but accuracy was only 0.637.
- **Choose the metric by the cost of errors.** For cancer screening, missing a malignant case is the costly mistake, so recall matters most.
- **Fit the scaler on training data only** to prevent data leakage.

## Getting Started

### Requirements

- Python 3.9+
- pandas
- scikit-learn
- Jupyter Notebook

```bash
pip install pandas scikit-learn notebook
```

### Run

1. Clone or download this project
2. Open the notebook:
   ```bash
   jupyter notebook ANN_first.ipynb
   ```
3. Run the cells from top to bottom. The first cell creates `breast_cancer.csv`.

## Project Structure

```
.
├── ANN_first.ipynb      # Main notebook (all steps and experiments)
├── breast_cancer.csv    # Dataset saved from scikit-learn
└── README.md
```

## Limitations

- Results come from a **single train/test split** on a small dataset, so small changes (one patient) can shift the scores. Cross-validation would give a more reliable estimate.
- The test set has only 64 malignant cases, so recall moves about 1.6 points per missed case.
- This is a learning project. It is **not a medical tool** and must not be used for real diagnosis.

## Next Steps

- Add cross-validation
- Try early stopping and regularization
- Compare with Random Forest
- Rebuild the ANN in Keras or PyTorch
