#  Neural Network Hyperparameter Optimization
### Architecture Design · Activation Functions · Learning Rate Tuning | Systematic ANN Study

**Skills demonstrated:** Python · Scikit-learn · Neural Networks (MLP/ANN) · Hyperparameter Tuning · Cross-Validation · Model Evaluation · Data Visualization · Healthcare Analytics · Business Decision-Making

---

## Project Overview

Conducted a systematic three-part study on Artificial Neural Network (ANN) hyperparameter optimization across two contrasting real-world datasets — a clean multi-class classification task (Iris) and a complex clinical dataset (Diabetes). Each study isolates one hyperparameter to measure its independent effect on convergence speed and model accuracy. Best configurations were validated using **5-fold cross-validation** and evaluated on held-out test sets with full Precision, Recall, and F1-score reporting. Every technical finding is translated into a business deployment recommendation.

---

## Business Problem

Machine learning teams in healthcare, finance, and retail regularly deploy ANNs — but default hyperparameter settings rarely deliver optimal results. Poor configuration choices create three measurable business problems:

| Problem | Business Impact |
|---|---|
| Sub-optimal learning rates cause slow convergence | Higher cloud compute costs — training jobs run longer than needed |
| Oversized networks on simple datasets | Slower API inference latency and higher memory overhead in production |
| Wrong activation function for the data type | 5–15% accuracy loss — directly reduces prediction reliability for business decisions |

**Solution:** A disciplined hyperparameter study quantifies which settings work for a given data profile — improving model accuracy and cutting training costs before any production code is written.

---

## Data Description

| Dataset | Task | Samples | Features | Target |
|---|---|---|---|---|
| **Iris** | Multi-class classification | 150 | 4 (sepal & petal dimensions) | 3 flower species |
| **Diabetes** | Risk classification | 442 | 10 clinical measurements | Disease progression → Low / Medium / High risk tiers |

- **Iris** represents clean, low-dimensional structured data — analogous to customer segmentation, product categorization, or churn prediction in business analytics
- **Diabetes** represents noisy, higher-complexity clinical data — analogous to insurance risk scoring, patient outcome prediction, or fraud detection pipelines
- All features scaled to [0, 1] using MinMaxScaler; diabetes target binarized into 3 clinically meaningful risk tiers (Low: 0–100, Medium: 100–175, High: 175+)
- No external download required — both datasets load automatically via `sklearn.datasets`

---

## Methodology

Three independent hyperparameter studies, each fixing all other variables while varying one:

- **Study 1 — Architecture (Neuron Count):** Hidden layer sizes of 10, 50, 100, and 150 neurons tested; loss convergence curves and 5-fold cross-validation accuracy recorded for both datasets
- **Study 2 — Activation Functions:** Logistic (sigmoid), Tanh, and ReLU compared on convergence speed, final training loss, and classification accuracy across both datasets
- **Study 3 — Learning Rate Sensitivity:** Five rates tested (0.001, 0.01, 0.05, 0.1, 0.2) using Adam optimizer; training stability, convergence status, and accuracy tracked to identify the safe operating window and divergence threshold
- **Final Best-Model Evaluation:** Optimal configuration per dataset trained on a 75/25 stratified split and evaluated on a held-out test set with confusion matrix (% format), per-class Precision/Recall/F1 bar chart, and full classification report
- **Cross-validation:** 5-fold CV applied across all three studies to confirm findings generalize beyond training data
- **Outputs saved:** Best models serialized as `.pkl`, all charts saved as `.png`, consolidated results exported to `.csv`

---

## Results

### Study 1 — Hidden Layer Neurons

| Dataset | Best Neurons | Accuracy | Key Finding |
|---|---|---|---|
| Iris | 100 | ~98% | Diminishing returns beyond 100 — adding 150 neurons gains <1% at 38% higher inference cost |
| Diabetes | 150 | ~72% | Complex clinical data continues improving with higher capacity |

### Study 2 — Activation Functions

| Dataset | Best Activation | Accuracy Gain vs Logistic | Key Finding |
|---|---|---|---|
| Iris | ReLU / Tanh | +5–8 pp | Logistic saturates early; Tanh converges fastest on balanced classes |
| Diabetes | ReLU | +10–12 pp | Gradient vanishing is severe for logistic on high-complexity data |

### Study 3 — Learning Rate

| Dataset | Optimal LR | Divergence Starts At | Key Finding |
|---|---|---|---|
| Iris | 0.01–0.05 | > 0.2 | Tolerates higher rates due to simple, linearly separable decision boundary |
| Diabetes | 0.01 | > 0.05 | Sensitive to high LR — training collapses to near-zero accuracy at lr=0.1 |

### Best Configuration — Held-Out Test Set

| Dataset | Best Config | Test Accuracy | 5-Fold CV Accuracy |
|---|---|---|---|
| **Iris** | 100 neurons · ReLU · lr=0.01 | ~96% | ~95% ± 2% |
| **Diabetes** | 150 neurons · ReLU · lr=0.01 | ~72% | ~70% ± 3% |

- **Delta (worst → best config per study):** Up to +15 pp on Diabetes from learning rate alone — confirming it is the highest-impact hyperparameter on complex data
- **Cross-validation gap vs training accuracy** remains narrow on both datasets, confirming models generalize and are not overfitting

---

## Business Recommendations

| Data Profile | Recommended ANN Config | Typical Use Case |
|---|---|---|
| Clean, structured, low-dimensional | 100 neurons · ReLU · lr=0.01 | Customer segmentation, churn prediction, product classification |
| Complex, noisy, high-dimensional | 150 neurons · ReLU · lr=0.001 | Clinical risk scoring, fraud detection, sensor data analytics |
| Real-time inference / edge deployment | 50 neurons · ReLU · lr=0.01 | Live recommendation APIs, streaming classification |

**Cost–accuracy tradeoff:** A 100-neuron ReLU network at lr=0.01 delivers the best balance for most business analytics tasks — 95%+ accuracy on structured data at 38% lower inference cost than a 150-neuron equivalent. Scale up neuron count only when 5-fold CV accuracy is still improving.

**Learning rate rule of thumb:** Always start at lr=0.001 or lr=0.01. Tune upward in log scale and stop at the first sign of validation loss instability. On complex datasets, lr > 0.05 is a reliable divergence risk threshold.

---

## Tools & Technologies

- **Language:** Python 3.10
- **Machine Learning:** Scikit-learn — MLPClassifier, cross_val_score, classification_report
- **Data Processing:** Pandas · NumPy
- **Visualization:** Matplotlib · Seaborn
- **Evaluation:** Confusion Matrix · Per-Class Precision/Recall/F1 · 5-Fold Cross-Validation
- **Model Persistence:** Pickle (`.pkl` serialization)
- **Environment:** Google Colab · Jupyter Notebook

---

## Project Files

```
neural-network-hyperparameter-optimization/
│
├── Neural_Network_Hyperparameter_Optimization_.ipynb   ← Full notebook (7 sections)
├── README.md                                           ← This file
│
└── Generated on notebook run
    ├── eda_overview.png                  ← EDA: scatter plots, correlation heatmap,
    │                                        feature distributions, risk tier bars
    ├── study1_neuron_count.png           ← Loss curves + accuracy bars by neuron count
    ├── study2_activation.png             ← Loss curves + accuracy bars by activation function
    ├── study3_learning_rate.png          ← Loss curves + accuracy bars by learning rate
    ├── best_model_evaluation.png         ← Training curve + confusion matrix +
    │                                        per-class F1 bars (held-out test set)
    ├── best_mlp_iris.pkl                 ← Saved best Iris model weights
    ├── best_mlp_diabetes.pkl             ← Saved best Diabetes model weights
    └── hyperparameter_study_results.csv  ← Consolidated findings: best/worst/delta
                                             across all 3 studies × 2 datasets
```

> **No dataset download needed.** Iris and Diabetes both load automatically via `sklearn.datasets`.

---

## How to Run

**Google Colab** *(recommended — no setup required, CPU is sufficient)*
1. Upload the notebook at [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Run all

**Local Jupyter**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
jupyter notebook Neural_Network_Hyperparameter_Optimization_.ipynb
```

---

## Author

**Aketch Okoth** · M.S. Business Analytics · Montclair State University  
*Actively seeking Data Analyst and Business Analyst roles in the United States.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=flat&logo=github&logoColor=white)](https://github.com/your-username)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:your-email@email.com)
