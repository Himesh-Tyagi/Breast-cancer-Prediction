# 🎗️ Breast Cancer Prediction Using Machine Learning

## 📌 Project Overview
This project uses machine learning techniques to build a predictive model that can classify whether a tumor is **malignant or benign** based on numeric input features. The model helps in early detection and supports healthcare decision-making by identifying patterns in medical data.

The project includes exploratory analysis, model training, evaluation, and a simple interface for predictions.

---

## 📁 Repository Structure

```
Breast-cancer-Prediction/
│
├── Breast Cancer.ipynb       # Main Jupyter Notebook with data analysis & model building
├── breast.pkl                # Saved machine learning model
├── breast.py                 # Prediction script / app interface
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📊 Dataset Description
The dataset consists of features computed from digitized images of breast masses. Each record includes attributes such as:
- Mean texture
- Mean perimeter
- Mean smoothness
- Mean compactness
- … and more

The target variable is:
- **Diagnosis:** Malignant or Benign

These features are used to train a classification model that predicts whether a tumor is likely cancerous.

---

## 🧠 Machine Learning Workflow
1. Load and explore the dataset  
2. Perform data preprocessing and cleaning  
3. Split data into training and testing sets  
4. Train classification models  
5. Evaluate model performance  
6. Save the best model (`breast.pkl`)  
7. Build an interface for predictions

---

## 🛠️ Tools & Technologies Used
- **Python**
- **Pandas & NumPy**
- **Scikit-Learn**
- **Jupyter Notebook**
- **Pickle / joblib**

---
---

## 📊 Key Results
- Developed a predictive classification model for breast cancer diagnosis
- Evaluated performance using accuracy and confusion matrix
- Saved a trained model for later use

---

## 🔮 Future Improvements
- Add feature scaling and hyperparameter tuning
- Implement more models (Random Forest, SVM, XGBoost)
- Create a web or GUI interface using Streamlit or Flask
- Deploy the app for public use

---

## 👨‍💻 Author
**Himesh Tyagi**  
Machine Learning & Data Analytics Enthusiast  


⭐ *If you find this project useful, don’t forget to star the repo!* ⭐
