# Regression Algorithms from Scratch

This repository demonstrates **Linear Regression** and **Logistic Regression** implemented from scratch using NumPy, with detailed training, evaluation, and visualisation. The goal is to provide a clear, educational look at how these foundational machine learning algorithms work under the hood for model fitting.

---

## Contents

- [`LR_code.ipynb`](https://github.com/KrishnaAggarwal2003/Regression-Algorithms-from-Scratch/blob/main/LR_code.ipynb): Linear Regression from scratch (for continuous targets)
- [`Logistic.ipynb`](https://github.com/KrishnaAggarwal2003/Regression-Algorithms-from-Scratch/blob/main/Logistic.ipynb): Logistic Regression from scratch (for binary classification)

---

## 1. Linear Regression (`LR_code.ipynb`)

### Overview

- **Data Generation:** Synthetic data is created with random features, coefficients, and Gaussian noise.
- **Model:** Implements multivariate linear regression using gradient descent.
- **Training:** Tracks cost (MSE) and R² score (accuracy) over epochs, with early stopping.
- **Evaluation:** Reports MSE, MAE, R², and visualizes predictions, residuals, and learned coefficients.

### Key Steps

1. **Data Creation:**  
   - Features (`X`), coefficients (`beta`), and noise are randomly generated.
   - Target (`Y`) is computed as a linear combination of features plus noise.

2. **Model Training:**  
   - Custom `LinearRegression` class with manual gradient descent.
   - Updates both coefficients and intercept.
   - Early stopping if the cost converges.

3. **Evaluation & Visualization:**  
   - Calculates MSE, MAE, and R² on the test set.
   - Plots predicted vs. actual values, residual distribution, and compares true vs. learned coefficients.

### Output obtained from Test-data

```
Epoch 0/500, Cost: 2.6241, Accuracy: -181.53%
...
Epoch 499/500, Cost: 0.0003, Accuracy: 99.97%

Range of Y data: -354.16 to 400.74, i.e. 754.9
Mean-squared Error: 3.6998
Mean-Absolute Error: 1.5285
R² score (Accuracy): 99.9647%
```
The model achieved excellent results. With a very high R² score of 99.9647%, it explains almost all of the variance in the Y data. The low Mean Squared Error (3.6998) and Mean Absolute Error (1.5285) relative to the wide range of the Y data (754.9) further confirm the model's high accuracy and small prediction errors.

![Plot](https://github.com/user-attachments/assets/7a8ec618-25b8-4a87-a9f8-71b34d528a58)


The graph clearly shows that the blue predicted points cluster tightly around the red "Ideal Fit" line. This strong alignment visually confirms the model's excellent performance, as indicated by the high R² score (99.9647%) and low error metrics previously discussed. The model's predictions are remarkably close to the true values across the entire range of data.

![Image](https://github.com/user-attachments/assets/295a6fcd-8ba1-4312-800d-b681ff293368)

This "Distribution of Residuals" histogram demonstrates the model's excellent performance by showing that its errors are normally distributed and centred around zero. This ideal distribution indicates that most predictions are highly accurate with small, unbiased errors, reinforcing the model's overall robustness and reliability.


![Image](https://github.com/user-attachments/assets/c9dcc4cc-be4c-4ff0-8558-b8668d7f6110)

This graph visually confirms the model's success in learning the underlying data relationships, as the "Learned Coefficients" (black bars) closely mirror the "Actual Coefficients" (blue bars) in both magnitude and direction for each feature variable. This strong alignment demonstrates the model's high accuracy in identifying the true influence of each feature on the target.

---
## 2. Logistic Regression (`Logistic.ipynb`)

### Overview

- **Data Generation:** Uses `make_classification` to create a synthetic binary classification dataset, with optional label noise.
- **Model:** Implements logistic regression with options for L1, L2, or combined regularization.
- **Training:** Uses gradient descent, tracks loss and accuracy, and supports early stopping.
- **Evaluation:** Reports classification metrics, confusion matrix, ROC curve, and visualises loss/accuracy curves.

### Key Steps

1. **Data Creation:**  
   - Features are standardised, and a bias term is added.
   - Optional label noise for realism.

2. **Model Training:**  
   - Custom `LogisticRegression` class with manual gradient descent.
   - Supports L1, L2, and combined regularization.
   - Tracks cost and accuracy per epoch.

3. **Evaluation & Visualization:**  
   - Classification report (precision, recall, f1-score, accuracy).
   - Plots: loss curve, accuracy curve, confusion matrix, ROC curve with AUC.

### Output from Test-data

```
Epoch 0/2000, Cost: 0.9063, Accuracy: 61.18%
...
Epoch 1999/2000, Cost: 0.5265, Accuracy: 88.98%
```
Classification Report
![Image](https://github.com/user-attachments/assets/aabc34f2-17ca-47be-b7e2-8bdb7de61fdf)

---

## 3. Visualizations

Both notebooks include rich visualisations:
- **Linear Regression:**  
  - Predicted vs. actual scatter plot  
  - Residuals histogram  
  - Bar plots comparing true and learned coefficients

- **Logistic Regression:**  
  - Loss and accuracy curves  
  - Confusion matrix  
  - ROC curve with AUC

---

## 4. How to Use

1. Clone the repository.
2. Open the notebooks in Jupyter or VS Code.
3. Run all cells to see data generation, model training, and result analysis.

---

## 5. Educational Value

- **No high-level model fitting:** All learning logic is implemented manually.
- **Step-by-step:** Each notebook walks through data creation, model logic, training, and evaluation.
- **Visualisation:** Plots help interpret model performance and learning dynamics.

---

## 6. Requirements

- Python 3.x
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- pandas

Install requirements with:
```bash
pip install numpy scikit-learn matplotlib seaborn tqdm pandas
```

---

## 7. License

This repository is licensed under the MIT License.
It is intended for educational and research purposes and demonstrates the inner workings of linear and logistic regression, including gradient descent, regularisation techniques, and performance evaluation metrics.

---
