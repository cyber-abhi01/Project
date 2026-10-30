import pandas as pd

import joblib



from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score







BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "student_data.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "student_model.pkl"





def load_dataset():

    """Load and validate the student dataset."""



    if not DATASET_PATH.exists():

        raise FileNotFoundError(

            f"Dataset not found: {DATASET_PATH}"

        )



    data = pd.read_csv(DATASET_PATH)



    required_columns = [

        "student_id",

        "study_hours",

        "attendance_percentage",

        "previous_marks",

        "final_marks",

    ]



    missing_columns = [

        column

        for column in required_columns

        if column not in data.columns

    ]



    if missing_columns:

        raise ValueError(

            f"Missing columns in dataset: {missing_columns}"

        )



    if data.empty:

        raise ValueError("The dataset is empty.")



    if data["student_id"].duplicated().any():

        raise ValueError("Duplicate student IDs found.")



    numeric_columns = required_columns[1:]



    for column in numeric_columns:

        data[column] = pd.to_numeric(

            data[column],

            errors="raise"

        )



    if data[numeric_columns].isnull().any().any():

        raise ValueError(

            "The dataset contains missing numerical values."

        )



    if not data[numeric_columns].map(

        lambda value: pd.notna(value) and abs(value) != float("inf")

    ).all().all():

        raise ValueError(

            "The dataset contains invalid numerical values."

        )



    if not data["study_hours"].between(0, 24).all():

        raise ValueError("Study hours must be between 0 and 24.")



    for column in [

        "attendance_percentage",

        "previous_marks",

        "final_marks",

    ]:

        if not data[column].between(0, 100).all():

            raise ValueError(

                f"{column} must be between 0 and 100."

            )



    if len(data) < 10:

        raise ValueError(

            "At least 10 student records are required."

        )



    return data





def train_model():

    """Train the model and evaluate its performance."""



    data = load_dataset()



    features = [

        "study_hours",

        "attendance_percentage",

        "previous_marks",

    ]



    X = data[features]

    y = data["final_marks"]



    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42

    )



    model = LinearRegression()

    model.fit(X_train, y_train)



    predictions = model.predict(X_test)



    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(

        y_test,

        predictions

    ) ** 0.5



    r2 = r2_score(y_test, predictions)



    

    MODEL_DIR.mkdir(parents=True, exist_ok=True)



    joblib.dump(model, MODEL_PATH)



    print("\nStudent Performance Prediction")

    print("-" * 35)

    print(f"Total records: {len(data)}")

    print(f"Training records: {len(X_train)}")

    print(f"Testing records: {len(X_test)}")

    print(f"Model saved to: {MODEL_PATH}")



    print("\nModel Evaluation")

    print("-" * 35)

    print(f"MAE:  {mae:.2f}")

    print(f"RMSE: {rmse:.2f}")

    print(f"R² Score: {r2:.4f}")



    print("\nSample Predictions")

    print("-" * 35)



    results = pd.DataFrame({

        "Actual Marks": y_test.values,

        "Predicted Marks": predictions.round(2)

    })



    print(results.head(5).to_string(index=False))



    print("\nModel training completed successfully!")





if __name__ == "__main__":

    try:

        train_model()

    except (FileNotFoundError, ValueError) as error:

        print(f"\nError: {error}")
