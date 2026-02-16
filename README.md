# 🌦️ Weather Classification Challenge (Domain Shift)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Score](https://img.shields.io/badge/F1--Macro-0.719-green)
![Rank](https://img.shields.io/badge/Rank-Top%20Tier-gold)

## 📖 Project Overview
This project focuses on building a robust **Weather Classification Model** capable of handling severe **Domain Shift**. 

The challenge involved training on standard, clear "Earth-like" images but testing on a dataset ("Alien/Mars") characterized by:
* **Inconsistent Lighting:** Extreme shadows and overexposure.
* **Color Shifts:** Tinted skies (red/green) that confuse standard color-based CNNs.
* **Constraint:** **No Pre-trained Models allowed** (e.g., no ImageNet weights).

I engineered a **Custom SE-ResNet** from scratch to solve this, prioritizing structural texture features over color.

---

## 🛠️ The Solution

### 1. Architecture: Custom SE-ResNet
Instead of standard transfer learning, I designed a specialized architecture:
* **ResNet Backbone:** Uses Skip Connections to solve the Vanishing Gradient problem, allowing for a deep feature extractor.
* **Squeeze-and-Excitation (SE) Blocks:** Integrated "Attention Mechanisms" that dynamically recalibrate channel importance. This allows the model to:
    * *Amplify* relevant texture features (e.g., vertical rain streaks).
    * *Suppress* misleading noise (e.g., alien sky colors).

### 2. Data Centric Engineering: CLAHE Preprocessing
Standard normalization failed due to the lighting variance in the test set. I implemented **CLAHE (Contrast Limited Adaptive Histogram Equalization)** in the **LAB Color Space**:
1.  Convert RGB image to LAB.
2.  Apply CLAHE to the **L (Lightness)** channel only.
3.  Convert back to RGB.
*Result: This "normalized" the lighting conditions, revealing hidden cloud/fog details in dark images without distorting color information.*

---

## 📊 Performance & Strategy

| Metric | Score |
| :--- | :--- |
| **F1-Macro** | **0.719** |
| **Rank** | **4th Place** |

**Key Techniques:**
* **Optimizer:** `Adam` with adaptive learning rates.
* **Regularization:** `Label Smoothing (0.1)` to prevent overconfidence on noisy data.
* **Inference:** **Test-Time Augmentation (TTA)**. Predictions are averaged across original and horizontally flipped images to ensure stability.

---

## 📂 Project Structure

```bash
├── dataset/
│   ├── train/          # 5 Classes: Cloudy, Foggy, Rainy, Shine, Sunrise
│   └── alien_test/     # The shifted domain test set
├── aws8.ipynb          # Main Jupyter Notebook (Training & Inference)
├── submission.csv      # Final predictions
└── README.md           # Project Documentation
