import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_INIT_AT_FORK"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
import json  # Add this import for JSON handling
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from gat import TrafficDataPreprocessor, GATTrafficPredictionModel, TrafficPredictor

MODEL_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat\\gat_model_20250319_144952.pth"
SCALER_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat\\scalers_20250319_144952.pkl"
SPEED_DATA_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\corrected_speed_data.xlsx"
DISTANCE_DATA_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\detector_distances.xlsx"
LAT_LONG_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\longlat.xlsx"
METRICS_PATH = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat\\metrics\\metrics_20250319_144555.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
CORS(app)

# Load lat/long data
lat_long_df = pd.read_excel(LAT_LONG_PATH)
print("Lat/Long DataFrame:\n", lat_long_df.head())
lat_long_dict = dict(zip(lat_long_df['detid'], lat_long_df[['lat', 'long']].to_dict('records')))
print("Lat/Long Dict Sample:", {k: lat_long_dict[k] for k in list(lat_long_dict.keys())[:5]})

# Load metrics data
with open(METRICS_PATH, 'r') as f:
    metrics_data = json.load(f)
print("Metrics Data Loaded:\n", metrics_data)

def load_model_and_preprocessor():
    preprocessor = TrafficDataPreprocessor(SPEED_DATA_PATH, DISTANCE_DATA_PATH)
    preprocessor.load_data()
    scalers = joblib.load(SCALER_PATH)
    preprocessor.scaler_X = scalers['scaler_X']
    preprocessor.scaler_y = scalers['scaler_y']
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    nfeat = state_dict['attentions.0.W'].shape[0]
    nhid = state_dict['attentions.0.W'].shape[1]
    nclass = state_dict['out_att.W'].shape[1]
    nheads = sum(1 for key in state_dict.keys() if key.startswith('attentions.') and key.endswith('.W'))
    num_nodes = state_dict['final_linear.weight'].shape[1] // nclass
    
    model = GATTrafficPredictionModel(
        nfeat=nfeat, nhid=nhid, nclass=nclass, num_nodes=num_nodes,
        dropout=0.6, alpha=0.2, nheads=nheads
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return preprocessor, model

preprocessor, model = load_model_and_preprocessor()
predictor = TrafficPredictor(model, preprocessor, device)

def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    print(f"Request method: {request.method}, Headers: {dict(request.headers)}")
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    start_detector = data.get('from')
    end_detector = data.get('to')
    print(f"Received /predict request: from={start_detector}, to={end_detector}")
    
    if not start_detector or not end_detector:
        return jsonify({"error": "Missing 'from' or 'to' parameters"}), 400
    
    try:
        result = predictor.predict_eta(start_detector, end_detector)
        print(f"Raw prediction result: {result}")
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        for path in result.get('alternative_paths', []):
            path['coordinates'] = [
                {
                    'detector': detector,
                    'latitude': lat_long_dict.get(detector, {}).get('lat', 0.0),
                    'longitude': lat_long_dict.get(detector, {}).get('long', 0.0)
                }
                for detector in path['path']
            ]
        
        serializable_result = convert_to_serializable(result)
        print(f"Serializable prediction result: {serializable_result}")
        return jsonify(serializable_result)
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/detectors', methods=['GET'])
def get_detectors():
    detectors = sorted(list(preprocessor.graph.nodes()))
    print("Detectors:", detectors)
    return jsonify({"detectors": detectors})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        # Add additional model info if needed
        metrics_response = {
            "model_loaded": True,
            "feature_count": model.nfeat if model else 0,  # From GAT model
            "sequence_length": preprocessor.sequence_length if preprocessor else 0,  # From preprocessor
            **metrics_data  # Spread the metrics from the JSON file
        }
        print("Metrics Response:", metrics_response)
        return jsonify(metrics_response)
    except Exception as e:
        print(f"Error in /metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)