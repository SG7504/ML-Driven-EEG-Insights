from fastapi import FastAPI, UploadFile, File
import pandas as pd
import uvicorn
import numpy as np
import json
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
from pyngrok import ngrok
import traceback
import ast

app = FastAPI()

# Load ANN weights
def load_ann_model():
    model_file = "C:\\Users\\KIIT\\Downloads\\ANN-without-power-2.csv"  # Adjusted for VS Code compatibility
    df = pd.read_csv(model_file)
    
    print("Raw ANN Model Data:\n", df.head())  # Debugging raw data
    
    # Extract weights only (ignoring biases for now)
    weight_rows = df[df['Type'].str.contains('weights', case=False, na=False)]
    weight_values = weight_rows['Values'].apply(ast.literal_eval)  # Convert string lists to actual lists
    
    # Convert to NumPy arrays while preserving layer structures
    weights = [np.array(w) for w in weight_values]
    print("Extracted Individual Weights Shapes:", [w.shape for w in weights])  # Debug shapes
    
    if len(weights) > 0:
        weights = weights  # Preserve as a list of arrays instead of stacking
    else:
        weights = []
    
    print("Final Weights Count:", len(weights))
    
    if not weights:
        print("Error: ANN Model weights are empty!")
    
    return weights

weights = load_ann_model()

def predict(input_data):
    input_array = np.array(input_data, dtype=float)  # Ensure numeric array
    print("Input Data:", input_array)  # Debugging input data
    print("Input shape:", input_array.shape)  # Debugging input shape
    
    if not weights:
        print("Error: Model weights are empty!")
        return []
    
    # Dynamically reshape input to match weight dimensions
    weight_matrix = weights[0]  # Assuming first layer is the input layer
    print("Using Weights Shape:", weight_matrix.shape)  
    
    if input_array.shape[1] != weight_matrix.shape[0]:
        print("Warning: Reshaping input data to match weight dimensions!")
        input_array = np.resize(input_array, (input_array.shape[0], weight_matrix.shape[0]))
    
    output = np.dot(input_array, weight_matrix.T)
    output = np.sign(output)  # Convert output to -1, 0, or 1
    
    print("Final Prediction Output:", output)  # Debugging output
    return output.tolist()

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict/")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print("Received Data:", df.head())  # Debugging input data
        print("Data shape:", df.shape)  # Debugging data shape
        
        # Ensure correct feature selection
        if df.shape[1] > 2:  # Adjust if extra columns exist
            df = df.iloc[:, -2:]  # Select the last two columns (numerical features)
        
        # Convert to float explicitly
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
        
        print("Processed Data Shape:", df.shape)  # Debugging new data shape
        print("Processed Data Types:\n", df.dtypes)  # Debugging data types
        print("Processed Data:\n", df.head())  # Print processed data
        
        predictions = predict(df.values)
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        error_msg = traceback.format_exc()
        print("Error Occurred:\n", error_msg)  # Print full error traceback
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/status/")
async def status():
    return {"status": "Backend is running"}

if __name__ == "__main__":
    # Set up Ngrok authentication
    ngrok.set_auth_token("2uAotzDPPTMvhkBlmRg4aCTwFDV_2pGUQq1xiJRci2SubwZCZ")
    
    # Start Ngrok tunnel
    public_url = ngrok.connect(8000).public_url
    print(f"Ngrok tunnel available at: {public_url}")
    
    # Start Uvicorn server normally for VS Code compatibility
    print("Starting Backend Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)