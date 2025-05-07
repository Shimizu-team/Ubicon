import pandas as pd
import numpy as np
import torch
from catboost import CatBoostClassifier
import joblib
import os
import json
from tqdm import tqdm

# Feature preparation function for prediction
def prepare_features_for_prediction(data_df, embeddings_dict, localization_df, protein_3di_dict=None):
    """
    Prepare feature vectors for prediction from E3-substrate data.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing E3-substrate pair information.
        embeddings_dict (dict): Dictionary mapping protein IDs to embedding vectors.
        localization_df (pd.DataFrame): DataFrame with protein localization scores.
        protein_3di_dict (dict, optional): Dictionary mapping protein IDs to their 3Di sequences.

    Returns:
        np.ndarray: Array of feature vectors.
    """
    features = []
    
    # List of 10 cellular compartments
    compartments = [
        'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 'Mitochondrion',
        'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome'
    ]
    
    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Preparing features"):
        e3_id = row['e3_uniprot_id']
        sub_id = row['substrate_uniprot_id']
        
        # Retrieve ESM embeddings
        if e3_id in embeddings_dict:
            e3_emb = embeddings_dict[e3_id].numpy() if isinstance(embeddings_dict[e3_id], torch.Tensor) else np.array(embeddings_dict[e3_id])
        else:
            # Use zero vector if embedding not found
            sample_key = list(embeddings_dict.keys())[0]
            sample_value = embeddings_dict[sample_key]
            if isinstance(sample_value, torch.Tensor):
                e3_emb = np.zeros_like(sample_value.numpy())
            else:
                e3_emb = np.zeros_like(np.array(sample_value))
            
        if sub_id in embeddings_dict:
            sub_emb = embeddings_dict[sub_id].numpy() if isinstance(embeddings_dict[sub_id], torch.Tensor) else np.array(embeddings_dict[sub_id])
        else:
            # Use zero vector if embedding not found
            sample_key = list(embeddings_dict.keys())[0]
            sample_value = embeddings_dict[sample_key]
            if isinstance(sample_value, torch.Tensor):
                sub_emb = np.zeros_like(sample_value.numpy())
            else:
                sub_emb = np.zeros_like(np.array(sample_value))
        
        # Handle 3Di features
        e3_3di = ''
        sub_3di = ''
        
        # If 3Di data is unavailable, try to retrieve it from protein_3di_dict.
        if protein_3di_dict and e3_id in protein_3di_dict:
            e3_3di = protein_3di_dict[e3_id]
        if protein_3di_dict and sub_id in protein_3di_dict:
            sub_3di = protein_3di_dict[sub_id]
            
        # Calculate 3Di frequency features
        e3_3di_counts = np.zeros(21)
        for char in e3_3di:
            if char in "DVPLSQCARNHGKMIFYWETX":
                idx = "DVPLSQCARNHGKMIFYWETX".index(char) + 1
                e3_3di_counts[idx] += 1
        e3_3di_freq = e3_3di_counts / max(len(e3_3di), 1)
        
        sub_3di_counts = np.zeros(21)
        for char in sub_3di:
            if char in "DVPLSQCARNHGKMIFYWETX":
                idx = "DVPLSQCARNHGKMIFYWETX".index(char) + 1
                sub_3di_counts[idx] += 1
        sub_3di_freq = sub_3di_counts / max(len(sub_3di), 1)
        
        # Localization features initialization
        e3_compartment_features = np.zeros(len(compartments))
        sub_compartment_features = np.zeros(len(compartments))
        
        # Populate E3 protein localization scores
        if localization_df is not None and e3_id in localization_df.index:
            for i, compartment in enumerate(compartments):
                if compartment in localization_df.columns:
                    try:
                        score = localization_df.at[e3_id, compartment]
                        if isinstance(score, (int, float)):
                            e3_compartment_features[i] = score
                    except:
                        pass
        
        # Populate substrate protein localization scores
        if localization_df is not None and sub_id in localization_df.index:
            for i, compartment in enumerate(compartments):
                if compartment in localization_df.columns:
                    try:
                        score = localization_df.at[sub_id, compartment]
                        if isinstance(score, (int, float)):
                            sub_compartment_features[i] = score
                    except:
                        pass
        
        # Concatenate all features into a single vector
        feature_vector = np.concatenate([e3_emb, sub_emb, e3_3di_freq, sub_3di_freq, e3_compartment_features, sub_compartment_features])
        features.append(feature_vector)
    
    return np.array(features)

# Function to process data in chunks
def process_chunk(data_df, model, embeddings_dict, localization_df, protein_3di_dict=None, batch_size=1000):
    """
    Process dataset in chunks and compute prediction scores.

    Parameters:
        data_df (pd.DataFrame): DataFrame with E3-substrate pairs.
        model (CatBoostClassifier): Trained classification model.
        embeddings_dict (dict): Dictionary of protein embeddings.
        localization_df (pd.DataFrame): DataFrame of localization information.
        protein_3di_dict (dict, optional): Dictionary of 3Di sequences.
        batch_size (int): Number of samples per batch.

    Returns:
        pd.DataFrame: Original DataFrame with added prediction scores.
    """
    # Get data length
    total_rows = len(data_df)
    
    
    # List of predicted results
    all_predictions = []
    
    # Split data for batch processing
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = data_df.iloc[start_idx:end_idx].copy()
        
        print(f"Processing batch: {start_idx}-{end_idx}/{total_rows}")
        
        # Prepare features
        X_batch = prepare_features_for_prediction(batch_df, embeddings_dict, localization_df, protein_3di_dict)
        
        # run predictions
        predicted_probs = model.predict_proba(X_batch)[:, 1]
        all_predictions.extend(predicted_probs)
    
    # Adding prediction scores to the original DataFrame
    result_df = data_df.copy()
    result_df['substrate_prediction_score'] = all_predictions
    
    return result_df

# Funtion to load only the model
def load_model(model_path):
    """
    Load only the CatBoost model without other resources.

    Parameters:
        model_path (str): Path to the CatBoost model file.

    Returns:
        CatBoostClassifier: Loaded model
    """
    print(f"Loading the model... {model_path}")
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        try:
            model = joblib.load(model_path.replace('.cbm', '.pkl'))
        except:
            raise Exception("Failed to load model. Please check model file.")
    
    return model
    
# Funtion to load model and data
def load_resources(model_path, embedding_path=None, localization_path=None, protein_3di_path=None):
    """
    Load model and data resources needed for prediction.

    Parameters:
        model_path (str): Path to the CatBoost model file.
        embedding_path (str, optional): Path to the embeddings data file.
        localization_path (str, optional): CSV file with localization data.
        protein_3di_path (str, optional): JSON file with 3Di sequences.

    Returns:
        tuple: (model, embeddings_dict, localization_df, protein_3di_dict)
    """
    print(f"Loading the model... {model_path}")
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        try:
            model = joblib.load(model_path.replace('.cbm', '.pkl'))
        except:
            raise Exception("Failed to load model. Please check model file.")
    
    # 埋め込みデータのロード
    embeddings_dict = {}
    if embedding_path:
        print(f"Loading embeddings data... {embedding_path}")
        try:
            embeddings_dict = torch.load(embedding_path)
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
    
    # 位置情報データのロード
    localization_df = None
    if localization_path:
        print(f"Loading localization data... {localization_path}")
        try:
            localization_df = pd.read_csv(localization_path, index_col=0)
        except Exception as e:
            print(f"Failed to load localization data: {e}")
    
    # 3Diデータのロード（オプション）
    protein_3di_dict = {}
    if protein_3di_path:
        print(f"Loading 3Di data... {protein_3di_path}")
        try:
            with open(protein_3di_path, "r") as f:
                protein_3di_dict = json.load(f)
        except Exception as e:
            print(f"Failed to load 3Di dat: {e}")
    
    return model, embeddings_dict, localization_df, protein_3di_dict

# キャリブレーションを適用する関数
def calibrate_scores(scores, calibration_model_path=None):
    """
    Apply calibration to prediction scores to generate Ubicon scores.

    Parameters:
        scores (np.ndarray): Array of predicted scores.
        calibration_model_path (str, optional): Path to calibration model (Isotonic Regression).
    Returns:
        np.ndarray: Calibrated scores (Ubicon scores).
    """
    try:
        # キャリブレーションモデルのロード
        if calibration_model_path and os.path.exists(calibration_model_path):
            calibration_model = joblib.load(calibration_model_path)
            # キャリブレーションの適用
            calibrated_scores = calibration_model.predict(scores)
            return calibrated_scores
        else:
            print("Calibration model not found. Returning original scores.")
            return scores
    except Exception as e:
        print(f"Error during calibration: {e}")
        return scores

# Extend the function for processing in chunks (with calibration support)
def process_chunk_with_calibration(data_df, model, embeddings_dict, localization_df, 
                                  protein_3di_dict=None, batch_size=1000, 
                                  calibration_model_path=None):
    """
    Process dataset in chunks, compute scores, and apply optional calibration.

    Parameters:
        data_df (pd.DataFrame): DataFrame with E3-substrate pairs.
        model (CatBoostClassifier): Trained classification model.
        embeddings_dict (dict): Dictionary of protein embeddings.
        localization_df (pd.DataFrame): DataFrame of localization information.
        protein_3di_dict (dict, optional): Dictionary of 3Di sequences.
        batch_size (int): Number of samples per batch.
        calibration_model_path (str, optional): Path to calibration model.

    Returns:
        pd.DataFrame: DataFrame with prediction and calibrated Ubicon scores.
    """
    result_df = process_chunk(data_df, model, embeddings_dict, localization_df, protein_3di_dict, batch_size)
    
    # Apply the calibration model if specified.
    if calibration_model_path:
        print(f"Applying calibration model: {calibration_model_path}")
        scores = np.array(result_df['substrate_prediction_score'])
        calibrated_scores = calibrate_scores(scores, calibration_model_path)
        result_df['ubicon_score'] = calibrated_scores
    
    return result_df