# 🧠 Artificial Neural Network for Customer Churn Prediction

Welcome to the **Customer Churn Prediction** project! This repository showcases the implementation of an Artificial Neural Network (ANN) to predict whether a customer is likely to churn based on various features. 🚀

---

## 📜 Problem Statement

Customer churn is a critical issue for businesses, especially in competitive industries like banking, telecom, and retail. Losing customers can significantly impact revenue and growth. The goal of this project is to build a machine learning model that predicts customer churn based on historical data, enabling businesses to take proactive measures to retain customers.

---

## 💡 Solution

This project uses an **Artificial Neural Network (ANN)** to classify customers as likely to churn or not. The solution includes:

- Preprocessing the dataset (encoding categorical variables, scaling features).
- Building and training an ANN model using TensorFlow/Keras.
- Hyperparameter tuning to optimize the model.
- Deploying the model using **Streamlit** for real-time predictions.

---

## 🛠️ Tech Stack

Here are the tools and technologies used in this project:

- **Python** 🐍
- **TensorFlow/Keras** for building and training the ANN.
- **scikit-learn** for preprocessing and hyperparameter tuning.
- **Streamlit** for creating an interactive web app.
- **Pandas** and **NumPy** for data manipulation.
- **Matplotlib** for visualizations.
- **TensorBoard** for monitoring training performance.

---

## 📊 Dataset

The dataset used in this project is the **Churn_Modelling.csv** file, which contains the following features:

- **CreditScore**: Customer's credit score.
- **Geography**: Customer's location (e.g., France, Germany, Spain).
- **Gender**: Customer's gender.
- **Age**: Customer's age.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Customer's account balance.
- **NumOfProducts**: Number of products the customer uses.
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: Customer's estimated salary.
- **Exited**: Target variable (1 = Churned, 0 = Not Churned).

---

## 🚀 Project Highlights

### 🔍 Data Preprocessing
- **Encoding**: Used `LabelEncoder` and `OneHotEncoder` for categorical variables.
- **Scaling**: Applied `StandardScaler` to normalize numerical features.

### 🧠 ANN Model
- **Architecture**:
  - Input Layer: Matches the number of features.
  - Hidden Layers: Two layers with 64 and 32 neurons, respectively.
  - Output Layer: Single neuron with a sigmoid activation function.
- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam with a learning rate of 0.01.
- **Metrics**: Accuracy.

### 🔧 Hyperparameter Tuning
- Used `GridSearchCV` with `scikeras.wrappers.KerasClassifier` to optimize:
  - Number of neurons.
  - Number of hidden layers.
  - Number of epochs.

GitHub Copilot
Here’s a well-structured and visually appealing README.md file for your project:

2️⃣ Install Dependencies
Make sure you have Python installed. Then, run:

3️⃣ Run the Streamlit App
4️⃣ Access the App
Open your browser and go to:

🧑‍💻 Skills Showcased
Machine Learning: Preprocessing, feature engineering, and model training.
Deep Learning: Building and optimizing an ANN.
Data Visualization: Using TensorBoard and Matplotlib.
Web Development: Deploying a model with Streamlit.
Hyperparameter Tuning: Using GridSearchCV for optimization.

📈 Results
Training Accuracy: ~95%
Validation Accuracy: ~92%
Churn Prediction: The model predicts churn probability with high accuracy.

### 🌐 Deployment
- Built an interactive **Streamlit** app for real-time predictions.
- Users can input customer details and get a churn probability score.

---

## 🖥️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MOHD-AFROZ-ALI/ANN-Classification--Churn
