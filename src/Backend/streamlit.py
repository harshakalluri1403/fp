import os
os.environ["OMP_NUM_THREADS"] = "1"  # Mitigate OpenMP conflicts
os.environ["KMP_INIT_AT_FORK"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import numpy as np
import joblib
import datetime
from gat import TrafficDataPreprocessor, GATTrafficPredictionModel, TrafficPredictor

# Paths to your model and scalers
MODEL_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat\\gat_model_20250319_144952.pth"
SCALER_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat\\scalers_20250319_144952.pkl"
SPEED_DATA_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\corrected_speed_data.xlsx"
DISTANCE_DATA_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\detector_distances.xlsx"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_preprocessor():
    # Initialize preprocessor
    preprocessor = TrafficDataPreprocessor(SPEED_DATA_PATH, DISTANCE_DATA_PATH)
    preprocessor.load_data()

    # Load scalers
    scalers = joblib.load(SCALER_PATH)
    preprocessor.scaler_X = scalers['scaler_X']
    preprocessor.scaler_y = scalers['scaler_y']

    # Load state dict
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # Infer model parameters from state_dict
    nfeat = state_dict['attentions.0.W'].shape[0]  # Input features (8)
    nhid = state_dict['attentions.0.W'].shape[1]   # Hidden features per head (64)
    nclass = state_dict['out_att.W'].shape[1]      # Output classes (4)
    # Count attention heads correctly
    nheads = sum(1 for key in state_dict.keys() if key.startswith('attentions.') and key.endswith('.W'))  # Should be 4
    final_linear_weight_shape = state_dict['final_linear.weight'].shape
    num_nodes = final_linear_weight_shape[1] // nclass  # Should be 20 (80 / 4)

    # Verify out_att.W input dimension
    expected_out_att_input = state_dict['out_att.W'].shape[0]  # 256
    calculated_input = nhid * nheads  # 64 * 4 = 256
    if calculated_input != expected_out_att_input:
        raise ValueError(f"Architecture mismatch: nhid ({nhid}) * nheads ({nheads}) = {calculated_input}, expected {expected_out_att_input}")

    # Create model with exact matching architecture
    model = GATTrafficPredictionModel(
        nfeat=nfeat,
        nhid=nhid,
        nclass=nclass,
        num_nodes=num_nodes,
        dropout=0.6,
        alpha=0.2,
        nheads=nheads  # Use 4 heads as per saved model
    )
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error(f"Failed to load state_dict: {str(e)}")
        st.write("State dict keys:", list(state_dict.keys()))
        raise
    
    model.to(device)
    model.eval()

    st.write(f"Loaded model with: nfeat={nfeat}, nhid={nhid}, nclass={nclass}, num_nodes={num_nodes}, nheads={nheads}")
    return preprocessor, model

def main():
    st.title("Traffic Prediction System")

    # Load model and preprocessor
    with st.spinner("Loading model..."):
        try:
            preprocessor, model = load_model_and_preprocessor()
            predictor = TrafficPredictor(model, preprocessor, device)
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return

    # Get available detectors
    detectors = sorted(list(preprocessor.graph.nodes()))

    # User input
    col1, col2 = st.columns(2)
    with col1:
        start_detector = st.selectbox("From Detector", detectors)
    with col2:
        end_detector = st.selectbox("To Detector", detectors)

    if st.button("Predict"):
        with st.spinner("Calculating prediction..."):
            try:
                result = predictor.predict_eta(start_detector, end_detector)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Prediction completed!")
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.write(f"Distance: {result['total_distance_km']:.2f} km")
                    st.write(f"Predicted Speed: {result['predicted_speed_kmh']:.2f} km/h")
                    st.write(f"ETA: {result['eta_minutes']:.1f} minutes")
                    st.write(f"Arrival Time: {result['estimated_arrival_time']}")
                    st.write(f"Path: {' -> '.join(map(str, result['path']))}")

                    # Weather information
                    st.subheader("Weather Conditions")
                    weather = result['weather_conditions']
                    st.write(f"Temperature: {weather['temperature']:.1f}°C")
                    st.write(f"Precipitation: {weather['precipitation']:.1f} mm")
                    st.write(f"Wind Speed: {weather['wind_speed']:.1f} km/h")
                    st.write(f"Humidity: {weather['humidity']:.1f}%")
                    st.write(f"Condition: {weather['description']}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()