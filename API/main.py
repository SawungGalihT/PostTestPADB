from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Energy Consumption Prediction API")

# Load model dan scaler
with open("random_forest_best.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_rf.pkl", "rb") as f:
    scaler = pickle.load(f)

# Skema input (harus cocok dengan fitur saat training)
class EnergyData(BaseModel):
    Total_Fossil_Fuels_Production: float
    Nuclear_Electric_Power_Production: float
    Total_Renewable_Energy_Production: float
    Total_Primary_Energy_Production: float
    Primary_Energy_Imports: float
    Primary_Energy_Exports: float
    Net_Energy_Movement: float
    Renewable_Energy_Ratio: float
    Fossil_Fuel_Dependency: float
    Nuclear_Energy_Share: float

# Preprocessing function
def preprocess_input(data: EnergyData):
    # Buat DataFrame dari input
    df = pd.DataFrame([{
        "Total Fossil Fuels Production": data.Total_Fossil_Fuels_Production,
        "Nuclear Electric Power Production": data.Nuclear_Electric_Power_Production,
        "Total Renewable Energy Production": data.Total_Renewable_Energy_Production,
        "Total Primary Energy Production": data.Total_Primary_Energy_Production,
        "Primary Energy Imports": data.Primary_Energy_Imports,
        "Primary Energy Exports": data.Primary_Energy_Exports,
        "Net Energy Movement": data.Net_Energy_Movement,
        "Renewable Energy Ratio": data.Renewable_Energy_Ratio,
        "Fossil Fuel Dependency": data.Fossil_Fuel_Dependency,
        "Nuclear Energy Share": data.Nuclear_Energy_Share
    }])

    # Scaling
    df_scaled = scaler.transform(df)
    return df_scaled

# Default route
@app.get("/")
def read_root():
    return {"message": "Energy Consumption Prediction API is running"}

# Predict route
@app.post("/predict")
def predict_energy(data: EnergyData):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    return {
        "prediction": float(prediction)
    }
