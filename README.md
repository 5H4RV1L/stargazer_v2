# 🌌 Stargazer AI — Exoplanet Classifier

A full-stack **Machine Learning web app** that trains and deploys an **XGBoost-based classification model** to predict exoplanet dispositions using CSV datasets.

Built with **Flask + Scikit-learn + XGBoost**, featuring a complete ML pipeline from data upload → preprocessing → training → prediction.

---

## 🚀 Features

* 📊 Upload labelled datasets (CSV)
* ⚙️ Automated preprocessing:

  * Missing value handling (median/mode)
  * Label encoding
  * Feature scaling (StandardScaler)
  * Outlier clipping (IQR method)
* 🧠 Feature engineering:

  * Radius ratios
  * Log transformations
  * Derived physics-based features
* ⚖️ Class balancing using **SMOTE**
* 🔁 5-fold **Stratified Cross Validation**
* 🤖 Model training with **XGBoost**
* 📈 Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix
* 💾 Model caching (saved using `joblib`)
* 🔮 Upload unlabelled data → get predictions
* 🎯 Individual row prediction viewer
* 🌐 Clean interactive UI (HTML + CSS)

---

## 🧠 ML Pipeline Overview

1. Upload labelled dataset
2. Preprocess data
3. Apply feature engineering
4. Balance classes using SMOTE
5. Train model (XGBoost)
6. Evaluate with cross-validation
7. Save model
8. Predict on new data

This pipeline is implemented inside `app.py` 

---

## 📁 Project Structure

```
├── app.py                # Flask backend + ML pipeline
├── requirements.txt     # Python dependencies
├── render.yaml          # Deployment config (Render)
├── templates/
│   ├── index.html       # Main UI
│   └── guide.html       # Dataset + pipeline guide
├── static/
│   ├── styles.css       # Styling
│   └── model_cache/     # Saved model
```

---

## 🛠️ Tech Stack

* **Backend:** Flask 
* **ML Libraries:**

  * Scikit-learn
  * XGBoost
  * Pandas / NumPy
  * Imbalanced-learn (SMOTE)
* **Frontend:** HTML, CSS
* **Deployment:** Render 

Dependencies listed in `requirements.txt` 

---

## ⚙️ Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/stargazer-ai.git
cd stargazer-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Visit: `http://127.0.0.1:5000`

---

## 📊 Dataset Requirements

Your CSV must include:

* Target column:

  ```
  koi_disposition
  ```

* Example features:

  ```
  koi_period
  koi_duration
  koi_depth
  koi_prad
  koi_srad
  ```

More details in `guide.html` 

---

## 🔮 Usage

### Training

1. Upload labelled CSV
2. Click **Train Model**
3. View metrics

### Prediction

1. Upload unlabelled CSV
2. Click **Run Predictions**
3. View results

---

## 💾 Model Persistence

* Model is automatically saved to:

  ```
  static/model_cache/model_bundle.joblib
  ```
* Reloaded on server restart
* No retraining required unless needed

(Implemented in `app.py`) 

---

## 🌐 Deployment (Render)

Configured via `render.yaml`:

```yaml
buildCommand: pip install -r requirements.txt
startCommand: gunicorn app:app
```



---

## 📸 UI Highlights

* Step-based ML workflow
* Drag & drop CSV upload
* Live model status indicator
* Confusion matrix visualization
* Prediction summaries

(UI defined in `index.html`) 

---

## 🧩 Future Improvements

* Hyperparameter tuning
* Model comparison (RF, SVM, etc.)
* Download predictions as CSV
* User authentication
* API endpoints

---

## 👨‍💻 Author

**Sharvil Sagalgile**

* Portfolio: https://sharvil.site
* GitHub: https://github.com/5H4RV1L

---

## 📜 License

This project is open-source and available under the MIT License.

---

⭐ If you found this useful, consider starring the repo!
