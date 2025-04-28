from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Titanic Survival Prediction API")

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Skema input
class Passenger(BaseModel):
    Name: str
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# Fungsi kategori umur
def categorize_age(age):
    if age < 18:
        return 'Child'
    elif 18 <= age < 60:
        return 'Adult'
    else:
        return 'Elderly'

def preprocess_input(data: Passenger):
    # Buat DataFrame dari input
    df = pd.DataFrame([{
        "Pclass": data.Pclass,
        "Sex": 1 if data.Sex.lower() == "male" else 0,
        "Age": data.Age,
        "SibSp": data.SibSp,
        "Parch": data.Parch,
        "Fare": data.Fare,
        "Embarked": data.Embarked.upper()
    }])


    df["AgeCategory"] = df["Age"].apply(categorize_age)
    df['AgeCategory'] = df['AgeCategory'].map({'Child': 0, 'Adult': 1,'Elder': 2}).astype(int)

    # One-hot encoding untuk 'Embarked' dan 'AgeCategory'
    df = pd.get_dummies(df, columns=["Embarked"])

    # Pastikan semua kolom dummy tersedia
    for col in ["Embarked_C", "Embarked_Q", "Embarked_S"]:
        if col not in df.columns:
            df[col] = 0

    # Urutkan kolom agar sesuai dengan model
    df = df[[
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare","AgeCategory",
        "Embarked_C", "Embarked_Q", "Embarked_S"
    ]]

    # Normalisasi
    df_scaled = scaler.transform(df)
    return df_scaled

@app.get("/")
def read_root():
    return {"message": "Titanic Prediction API is running"}

# Endpoint prediksi
@app.post("/predict")
def predict_survival(data: Passenger):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    result = "Survived" if prediction == 1 else "Not Survived"
    return {
        "name": data.Name,
        "prediction": int(prediction),
        "result": result
    }