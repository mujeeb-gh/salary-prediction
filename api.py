from flask import Flask, request, jsonify, render_template
import pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

app = Flask(__name__)

# Load data and model
clean_data = "data/Clean_Salary_Data.csv"
rf_model = joblib.load("models/random_forests_model.pkl")

# Input function
genders= ["Male", "Female", "Other"]
education_level_options = ["High School", "Bachelor's Degree", "Master's Degree", "PhD"]
job_titles = pd.read_csv(clean_data)["Job Title"].unique().tolist()

def input_features(data):
    age = data["age"]
    gender = data["gender"].capitalize()
    educationLevel = data["educationLevel"]
    jobTitle = data["jobTitle"]
    years = data["yearsOfExperience"]
    
    if age < 21 or age > 62:
        raise ValueError("Age must be a number between 21 and 62.")
    if gender not in genders:
        raise ValueError("Gender must be Male, Female, or Other")
    if educationLevel not in education_level_options:
        raise ValueError("Education Level must be one of the specified options.")
    if jobTitle not in job_titles:
        raise ValueError("Job Title must be one of the specified options.")
    if years < 0 or years > 34:
        raise ValueError("Years of Experience must be between 0 and 34 years.")
    
    return age, gender, educationLevel, jobTitle, years


# Define API route
@app.route('/predict', methods = ['POST'])
def predict_salary():
    model_data = pd.read_csv(clean_data)
    global df
    df = model_data.drop(["Unnamed: 0", "Salary", "Age Group"], axis=1)

    data = request.json # Get JSON input from user
    
    try:
        # Use input_features function to extract data from JSON
        age, gender, educationLevel, jobTitle, years = input_features(data)
        
        # Create input dataframe
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [educationLevel],
            "Job Title": [jobTitle],
            "Years of Experience": [years]
        })
        input_row = input_df.iloc[0]
        # df = df.append(input_row, ignore_index= True)
        df = pd.concat([df, input_row.to_frame().T], ignore_index =True)
        
        # Load the model's preprocessing steps
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(drop="first", sparse_output=False)
        
        # Preprocess the input dataframe
        df["Education Level"] = label_encoder.fit_transform(df["Education Level"])
        onehot_encoded = onehot_encoder.fit_transform(df[["Gender", "Job Title"]])
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(["Gender", "Job Title"]))
        df = pd.concat([df, onehot_df], axis=1)
        df = df.drop(["Gender", "Job Title"], axis=1)
        
        # Prediction
        predicted_salary = rf_model.predict(df.tail(1))[0]
        
        # Return result as a JSON
        response = {
            "age": age,
            "gender": gender,
            "educationLevel": educationLevel,
            "jobTitle": jobTitle,
            "yearsOfExperience": years,
            "predictedSalary": predicted_salary
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
  app.run(debug=True, port=8000)