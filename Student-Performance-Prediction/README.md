# Student Performance Prediction Using Machine Learning

## About the Project

This is a Machine Learning project that predicts a student's expected final exam marks based on their daily study hours, attendance percentage, and previous exam marks.

The main idea behind this project is to understand how different academic factors can be used to estimate student performance.

I have used Python to build the project and Streamlit to create a simple web application where users can enter student details and get predicted marks.

## Technologies Used

- Python – Main programming language
- Pandas – To read and manage the dataset
- NumPy – For numerical operations
- Scikit-learn – To train and evaluate the ML model
- Matplotlib and Seaborn – To create graphs
- Streamlit – To build the web interface
- Joblib – To save and load the trained model

## Dataset

The project uses a CSV file named `student_data.csv`, which contains 50 sample student records.

The dataset includes the following information:

- Student ID
- Daily study hours
- Attendance percentage
- Previous exam marks
- Final exam marks

The dataset contains synthetic sample data created for this project. It does not contain actual student records.

## How the Project Works

First, the program loads the dataset and checks whether the required columns and values are available.

Next, the dataset is divided into training and testing data. A Linear Regression model is trained using study hours, attendance, and previous exam marks.

After training, the model is evaluated using MAE, RMSE, and R² score to understand how well it performs on the test data.

The trained model is saved and used by the Streamlit application to predict expected final marks for new inputs.

## Features

- Predicts expected final exam marks.
- Accepts study hours, attendance, and previous marks as input.
- Displays a performance category based on the predicted score.
- Shows basic information about the dataset.
- Displays graphs for study hours, attendance, and final marks.
- Allows users to view and download the sample dataset.
- Shows model evaluation results during training.

## Project Structure

```text
Student-Performance-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── student_data.csv
│
├── models/
│   └── student_model.pkl
│
└── outputs/