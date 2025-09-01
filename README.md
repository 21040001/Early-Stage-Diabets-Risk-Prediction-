Tamam, işte projeyi anlatan, kullanıma hazır bir `README.md` dosyasının içeriği. Bu içeriği doğrudan projenizin kök dizinindeki `README.md` dosyasına kopyalayabilirsiniz.

```markdown
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
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Example `requirements.txt`:*
```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
tensorflow==2.12.0  # For the Neural Network
```

### 3. Run the Jupyter Notebook
The main analysis is conducted in the Jupyter Notebook `diabetes_risk_classification.ipynb`.
```bash
jupyter notebook diabetes_risk_classification.ipynb
```

### 4. Run a Python Script (Alternative)
Alternatively, you can run the provided Python script.
```bash
python main.py
```

---

## 📊 Results

The performance of each model was evaluated using standard metrics. The results are summarized below:

| Model                | Accuracy | Precision | Recall (Sensitivity) | F1-Score | AUC   |
| -------------------- | -------- | --------- | -------------------- | -------- | ----- |
| Naive Bayes          | 0.91     | 0.90      | 0.90                 | 0.90     | 0.95  |
| K-NN                 | 0.93     | 0.87      | 0.90                 | 0.88     | 0.98  |
| Logistic Regression  | 0.92     | 0.92      | 0.90                 | 0.91     | 0.97  |
| Decision Tree        | 0.98     | 0.97      | 0.99                 | 0.98     | 0.98  |
| **Random Forest**    | **0.99** | **0.99**  | **0.99**             | **0.99** | **1.00** |
| Linear Regression    | 0.60     | 0.92      | 0.90                 | 0.91     | 0.98  |
| Artificial Neural Network | 0.97  | 0.96      | 0.98                 | 0.97     | 1.00  |

### 📈 Key Findings:
- **Random Forest** and **Artificial Neural Network** achieved the best overall performance.
- **Linear Regression** is unsuitable for this binary classification task, yielding the lowest accuracy (60%).
- The high performance of tree-based models (Decision Tree, Random Forest) suggests strong, learnable patterns in the symptom data.

---

## 🧠 Model Performance Visualization

The project includes visualizations for:
- **Confusion Matrices** for each model.
- **ROC Curves** and AUC scores.
- **Feature Importance** charts (for Decision Tree/Random Forest).

*(These plots are generated automatically in the provided code.)*

---

## 👥 Authors

- **Kağan Güner** - [kagan.guner@gazi.edu.tr](mailto:kagan.guner@gazi.edu.tr)
- **Davronbek Abdurazzokov** - [23181616403@gazi.edu.tr](mailto:23181616403@gazi.edu.tr)

**Affiliation:** Department of Computer Engineering, Faculty of Technology, Gazi University, Ankara, Turkey.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to Gazi University for supporting this research.
- Dataset providers and UCI Machine Learning Repository.
- The open-source community for providing invaluable tools and libraries.

---

## 🔗 Citation

If you use this code or report in your research, please cite the original work:

```bibtex
@article{guner2023diabetes,
  title={Classification of Early-Stage Diabetes Risk with Machine Learning},
  author={Guner, Kagan and Abdurazzokov, Davronbek},
  journal={Gazi University Data Science Research Project},
  year={2023}
}
```
```

**Not:** Bu README, projenizin bir GitHub veya GitLab sayfasında görünecek şekilde tasarlanmıştır. `your-username` ve repository linkini kendi bilgilerinizle değiştirmeyi unutmayın. Ayrıca, `requirements.txt` dosyasındaki kütüphane versiyonları örnektir, kendi ortamınıza göre güncelleyin.
