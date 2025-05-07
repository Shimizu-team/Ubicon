import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import json
import joblib

# Import from src
sys.path.append("src")
from score_utils import load_model, process_chunk, calibrate_scores

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict interaction score for a specific E3-substrate pair using Ubicon")
    
    # Required IDs
    parser.add_argument("--e3_id", type=str, required=True,
                        help="UniProt ID of the E3 ligase")
    parser.add_argument("--substrate_id", type=str, required=True,
                        help="UniProt ID of the substrate")
    
    # Configuration (optional)
    parser.add_argument("--config", type=str, default="config/default_paths.json",
                        help="Path to configuration file with paths to feature files")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration file with paths to feature files."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        # Create default configuration
        default_config = {
            "e3_embedding": "examples/E3_feature_embedding.pt",
            "substrate_embedding": "examples/Sub_feature_embedding.pt",
            "e3_structure": "examples/E3_structure_embed.json",
            "substrate_structure": "examples/Sub_structure_embed.json",
            "e3_location": "examples/E3_location_embedding.csv",
            "substrate_location": "examples/Sub_location_embedding.csv",
            "model_path": "models/final_catboost_model.cbm",
            "calibration_path": "models/isotonic_calibration_model.pkl"
        }
        return default_config

def load_embeddings(e3_path, substrate_path):
    """Load embeddings from files."""
    print(f"Loading embeddings...")
    try:
        e3_embeddings = torch.load(e3_path)
        substrate_embeddings = torch.load(substrate_path)
        combined_embeddings = {**e3_embeddings, **substrate_embeddings}
        return combined_embeddings
    except Exception as e:
        raise ValueError(f"Failed to load embeddings: {e}")

def load_structure_data(e3_path, substrate_path):
    """Load structure data from files."""
    print(f"Loading structure data...")
    try:
        with open(e3_path, 'r') as f:
            e3_structure = json.load(f)
        with open(substrate_path, 'r') as f:
            substrate_structure = json.load(f)
        combined_structure = {**e3_structure, **substrate_structure}
        return combined_structure
    except Exception as e:
        raise ValueError(f"Failed to load structure data: {e}")

def load_location_data(e3_path, substrate_path):
    """Load location data from files."""
    print(f"Loading location data...")
    try:
        e3_location = pd.read_csv(e3_path, index_col=0)
        substrate_location = pd.read_csv(substrate_path, index_col=0)
        combined_location = pd.concat([e3_location, substrate_location])
        return combined_location
    except Exception as e:
        raise ValueError(f"Failed to load location data: {e}")

def main():
    """Main function to predict E3-substrate interaction score."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create pairs dataframe
    pairs_df = pd.DataFrame([{
        'e3_uniprot_id': args.e3_id,
        'substrate_uniprot_id': args.substrate_id,
        'e3_name': args.e3_id,
        'substrate_name': args.substrate_id
    }])
    
    # Load features
    try:
        embeddings = load_embeddings(config["e3_embedding"], config["substrate_embedding"])
        structure_data = load_structure_data(config["e3_structure"], config["substrate_structure"])
        location_data = load_location_data(config["e3_location"], config["substrate_location"])
    except Exception as e:
        print(f"Error: Failed to load feature files: {e}")
        return 1
    
    # Check if IDs exist in the feature files
    if args.e3_id not in embeddings:
        print(f"Error: E3 ligase ID '{args.e3_id}' does not exist in the feature files")
        return 1
    if args.substrate_id not in embeddings:
        print(f"Error: Substrate ID '{args.substrate_id}' does not exist in the feature files")
        return 1
        
    # Load model
    model_path = config.get("model_path", "models/final_catboost_model.cbm")
    print(f"Loading model from {model_path}")
    
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        return 1
    
    # Calculate predictions
    print(f"Calculating scores for E3: {args.e3_id} and Substrate: {args.substrate_id}...")
    
    try:
        results_df = process_chunk(
            pairs_df, 
            model, 
            embeddings, 
            location_data, 
            structure_data
        )
    except Exception as e:
        print(f"Error: Failed to calculate scores: {e}")
        return 1
    
    # Apply calibration if available
    calibration_path = config.get("calibration_path", "models/isotonic_calibration_model.pkl")
    if os.path.exists(calibration_path):
        try:
            print(f"Applying calibration model...")
            scores = np.array(results_df['substrate_prediction_score'])
            calibrated_scores = calibrate_scores(scores, calibration_path)
            results_df['ubicon_score'] = calibrated_scores
            final_score = calibrated_scores[0]
        except Exception as e:
            print(f"Warning: Calibration failed: {e}")
            final_score = results_df['substrate_prediction_score'][0]
    else:
        final_score = results_df['substrate_prediction_score'][0]
    
    # Print result
    print("\n===== Prediction Result =====")
    print(f"E3 ligase: {args.e3_id}")
    print(f"Substrate: {args.substrate_id}")
    print(f"Ubicon score: {final_score:.6f}")
    print("====================\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())