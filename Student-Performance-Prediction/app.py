import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent


DATASET_PATH = BASE_DIR / "dataset" / "student_data.csv"



MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"


st.set_page_config(

    page_title="Student Performance Prediction",

    page_icon="🎓",

    layout="wide"
)



@st.cache_resource

def load_model():


    if not MODEL_PATH.exists():
        return None
    

    return joblib.load(MODEL_PATH)



@st.cache_data
def load_dataset():

    if not DATASET_PATH.exists():

        return None



    return pd.read_csv(DATASET_PATH)





def get_performance_category(marks):

    if marks >= 80:

        return "Excellent"

    elif marks >= 65:

        return "Good"

    elif marks >= 40:

        return "Average"

    else:

        return "Needs Improvement"







st.title("🎓 Student Performance Prediction")

st.write(

    "A Machine Learning project that estimates a student's "

    "final exam marks using study hours, attendance, "

    "and previous exam performance."

)



st.info(

    "This project uses synthetic sample data for educational "

    "demonstration. Predictions are estimates, not guaranteed results."

)







model = load_model()

data = load_dataset()





if model is None:

    st.error(

        "Trained model not found. Please run "

        "`python train_model.py` from the project folder first."

    )

    st.stop()





if data is None:

    st.error("Dataset not found in the dataset folder.")

    st.stop()







st.sidebar.header("About the Project")

st.sidebar.write(

    "This application uses Linear Regression to predict "

    "expected final exam marks."

)



st.sidebar.markdown("---")

st.sidebar.write("**Input Features**")

st.sidebar.write("• Daily study hours")

st.sidebar.write("• Attendance percentage")

st.sidebar.write("• Previous exam marks")






st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Students", len(data))

with col2:
    st.metric(
        "Average Final Marks",
        f"{data['final_marks'].mean():.2f}"
    )

with col3:
    st.metric(
        "Average Attendance",
        f"{data['attendance_percentage'].mean():.1f}%"
    )


st.markdown("---")



st.subheader("Predict Student Performance")

st.write(
    "Enter the student's academic details to estimate "
    "their final exam score."
)


with st.form("prediction_form"):
    


    col1, col2, col3 = st.columns(3)
    


    with col1:
        
        study_hours = st.number_input(
            
            "Daily Study Hours",
            
            min_value=0.0,
            
            max_value=24.0,
            
            value=3.0,
            
            step=0.5
            
        )
        


    with col2:
        
        attendance = st.number_input(
            
            "Attendance Percentage",
            
            min_value=0.0,
            
            max_value=100.0,
            
            value=75.0,
            
            step=1.0
            
        )
        


    with col3:
        
        previous_marks = st.number_input(
            
            "Previous Exam Marks",
            
            min_value=0.0,
            
            max_value=100.0,
            
            value=60.0,
            
            step=1.0
            
        )
        


    predict_button = st.form_submit_button(
        
        "Predict Final Marks",
        
        use_container_width=True
        
    )
    




if predict_button:
    


    input_data = pd.DataFrame({
        
        "study_hours": [study_hours],
        
        "attendance_percentage": [attendance],
        
        "previous_marks": [previous_marks]
        
    })

    try:
        prediction = float(model.predict(input_data)[0])

        if not pd.notna(prediction):
            raise ValueError("The model returned an invalid prediction.")

       
        predicted_marks = max(0.0, min(100.0, prediction))

        category = get_performance_category(predicted_marks)

        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Final Marks",
                f"{predicted_marks:.2f} / 100"
            )

        with col2:
            st.metric(
                "Performance Category",
                category
            )

        st.progress(predicted_marks / 100)

        st.caption(
            "Performance categories are simple project-defined "
            "labels. They are not scientifically validated classifications."
        )

    except (ValueError, TypeError) as error:
        st.error(f"Prediction could not be completed: {error}")



st.markdown("---")

st.subheader("Student Dataset")



with st.expander("View Dataset"):

    st.dataframe(

        data,

        use_container_width=True,

        hide_index=True

    )



    st.download_button(

        "Download Dataset CSV",

        data=data.to_csv(index=False).encode("utf-8"),

        file_name="student_data.csv",

        mime="text/csv"

    )







st.markdown("---")

st.subheader("Data Analysis and Visualizations")



chart_col1, chart_col2 = st.columns(2)



with chart_col1:

    st.write("**Study Hours vs Final Marks**")



    fig, ax = plt.subplots(figsize=(7, 4))



    sns.scatterplot(

        data=data,

        x="study_hours",

        y="final_marks",

        ax=ax

    )



    ax.set_xlabel("Daily Study Hours")

    ax.set_ylabel("Final Marks")

    ax.set_title("Study Hours and Exam Performance")



    st.pyplot(fig)

    plt.close(fig)





with chart_col2:

    st.write("**Attendance vs Final Marks**")



    fig, ax = plt.subplots(figsize=(7, 4))



    sns.scatterplot(

        data=data,

        x="attendance_percentage",

        y="final_marks",

        ax=ax

    )



    ax.set_xlabel("Attendance Percentage")

    ax.set_ylabel("Final Marks")

    ax.set_title("Attendance and Exam Performance")



    st.pyplot(fig)

    plt.close(fig)





st.write("**Distribution of Final Marks**")



fig, ax = plt.subplots(figsize=(9, 4))



sns.histplot(

    data=data,

    x="final_marks",

    bins=10,

    kde=True,

    ax=ax

)



ax.set_xlabel("Final Marks")

ax.set_ylabel("Number of Students")

ax.set_title("Distribution of Student Final Marks")



st.pyplot(fig)

plt.close(fig)







st.markdown("---")

st.subheader("About the Machine Learning Model")



st.write(

    """

    The application uses Linear Regression, a supervised machine

    learning algorithm. The model learns relationships between

    study hours, attendance, previous marks, and final marks.



    Its performance depends on the quality and size of the dataset.

    The current dataset contains synthetic examples for demonstration,

    so its predictions should not be used for actual academic decisions.

    """

)



st.caption("Student Performance Predictions")
