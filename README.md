# 🛒 E-commerce Conversion Predictor (Streamlit + Machine Learning)

This project is an **e-commerce customer conversion prediction system** that analyzes clickstream data to predict whether a user will convert (reach checkout/purchase pages) based on their browsing behavior.

## 📊 Project Overview

The system predicts customer conversion likelihood using **clickstream data** from an online clothing store. It uses machine learning to analyze user interactions and predict if they'll proceed to checkout (pages 4-5) or drop off earlier.

## 🚀 Key Features

- **Conversion Prediction**: Predicts if a customer will convert based on browsing behavior
- **Interactive Dashboard**: User-friendly Streamlit web interface
- **Model Training & Evaluation**: Complete ML pipeline from training to evaluation
- **Real-time Predictions**: Get instant conversion predictions with user inputs
- **Performance Metrics**: Model accuracy evaluation and reporting

## 🛠️ Technology Stack

- **Python 3.11+**
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library
- **pandas & numpy** - Data manipulation
- **joblib** - Model serialization
- **Jupyter Notebook** - Data analysis

## 🧾 How to Run this Project
```bash
- git clone https://github.com/keshavjatt/Clickstream-Customer-Conversion.git 
- cd Clickstream-Customer-Conversion
- pip install -r requirements.txt
- python train_model.py
- python evaluate.py
- python -m streamlit run app.py