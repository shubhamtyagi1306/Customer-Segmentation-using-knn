# 🧠 Customer Segmentation using K-Nearest Neighbors (KNN)

This project uses the K-Nearest Neighbors (KNN) algorithm to classify customers into segments based on their demographics and spending behavior. We go through building a base model, performing hyperparameter tuning, and analyzing the results with a focus on model performance and generalization.

---

## 📌 Problem Statement

We aim to predict a customer’s **spending behavior category** (Low, Medium, High) based on:
- Gender
- Age
- Annual Income

We transformed this into a **classification problem** by segmenting the `Spending Score` into 3 categories:
- `0`: Low spender
- `1`: Medium spender
- `2`: High spender

---

## 📊 Dataset

**Mall Customer Segmentation Data**  
📂 Download from Kaggle: [Click here](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation/data?select=Mall_Customers.csv)

**Features:**
- `Gender` (converted to 0/1)
- `Age`
- `Annual Income (k$)`
- `Spending Score (1–100)` ➜ Used to create target class: `Segment`

---

## ✅ Project Workflow

1. **Data Cleaning & Preprocessing**
   - Encoded categorical variables
   - Normalized features using `StandardScaler`
   - Created a `Segment` target variable by binning `Spending Score`

2. **Base KNN Model**
   - Used default `k=5`
   - Achieved high test accuracy: **82.5%**
   - However, this could be **overfitting** to the current train-test split

3. **Hyperparameter Tuning (GridSearchCV)**
   - Tuned `n_neighbors`, `weights`, and `distance metric`
   - Best parameters: e.g., `k=4`, `weights='uniform'`, `metric='euclidean'`
   - Final test accuracy: **77.5%**
   - Accuracy slightly dropped but model **generalizes better on unseen data**

4. **Elbow Method Visualization**
   - Confirmed `k=4` is near the optimal elbow point

---

## 📈 Results & Interpretation

| Model                  | Accuracy | Comment                        |
|------------------------|----------|--------------------------------|
| Base KNN               | 82.5%    | High accuracy but overfitting |
| Tuned KNN (CV-based)   | 77.5%    | More stable, better generalization |

- The **base model** may have overfitted the test data (especially due to small dataset size).
- The **tuned model**, despite lower accuracy, is **less biased by specific train-test splits** and performs **more reliably** when predicting **new or unseen data**.

---

## 🧪 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

---

## 📘 Key Learnings

- Always validate model performance using **cross-validation**, not just one train-test split.
- **Higher accuracy ≠ better model**-generalization matters more.
- KNN is sensitive to scaling and the choice of `k`, so visualization (like Elbow Method) is helpful.

---

