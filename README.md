# Classification of Early-Stage Diabetes Risk with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)](https://scikit-learn.org/stable/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

This repository contains the code and report for a research project that aims to classify the risk of early-stage diabetes using various machine learning algorithms. The study is conducted as part of a Data Science research project at Gazi University.

---

## 📖 Abstract

Diabetes is a widespread and potentially fatal chronic disease. Early diagnosis and intervention are crucial to mitigate its adverse effects and improve patients' quality of life. This project leverages machine learning to predict diabetes risk based on symptom data, eliminating the need for initial medical procedures like blood tests.

We used the **Early Stage Diabetes Risk Prediction Dataset**, which includes 16 symptom-based features from 520 individuals. Seven different machine learning models were trained and evaluated. The **Random Forest** algorithm achieved the highest performance with **99% accuracy, precision, recall, and F1-score**.

---

## 🗂️ Dataset

**Source:** [UCI Machine Learning Repository - Early Stage Diabetes Risk Prediction Dataset](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset)

**Features:**
- **17 attributes** (16 symptoms + 1 target variable)
- **520 instances**
- Attributes include: `Age`, `Gender`, `Polyuria`, `Polydipsia`, `sudden weight loss`, `weakness`, `Polyphagia`, `Genital thrush`, `visual blurring`, `Itching`, `Irritability`, `delayed healing`, `partial paresis`, `muscle stiffness`, `Alopecia`, `Obesity`
- **Target:** `Class` (Positive / Negative for diabetes)

**Preprocessing:**
- Categorical values (`Yes`/`No`, `Male`/`Female`, `Positive`/`Negative`) were converted to binary (`1`/`0`).
- The dataset was split into **80% training** and **20% testing** using `train_test_split`.

---

## 🤖 Models Implemented

We implemented and compared the following seven machine learning algorithms:

1.  **Naive Bayes**
2.  **Logistic Regression**
3.  **Decision Tree**
4.  **Random Forest**
5.  **K-Nearest Neighbors (K-NN)**
6.  **Linear Regression** (for baseline comparison)
7.  **Artificial Neural Network (ANN)** with 2 hidden layers

---

## ⚙️ Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/early-diabetes-risk-prediction.git
cd early-diabetes-risk-prediction
