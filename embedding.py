import torch
import pandas as pd
import os
import logging
import argparse

from src.trainer_lora import Trainer
from src.model_setting import create_embedding_model
from src.dataset_lora import create_embedding_loaders, convert_fasta_to_pt

# Logging configuration (displaying filename and line number)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

def fasta_to_dict(fasta_path):
    """
    Convert a FASTA file to a dictionary with {ID: sequence}.
    
    Parameters:
        fasta_path (str): Path to the FASTA file.
    
    Returns:
        dict: Dictionary mapping sequence IDs to sequences.
    """
    seq_dict = {}
    with open(fasta_path, 'r') as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    seq_dict[seq_id] = ''.join(seq_lines)
                seq_id = line[1:]  # remove '>'
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id is not None:
            seq_dict[seq_id] = ''.join(seq_lines)  # last sequence
    return seq_dict

def Embedding(E3_seq_dict, Sub_seq_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Create a directory using the current timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    timestamp = timestamp + '_embedding'
    result_dir = os.path.join('results', timestamp)
    
    logger.info(f'Using device: {device}')
    os.makedirs(result_dir, exist_ok=True)   
    logger.info(f'Results will be saved in: {result_dir}')
        

    # Create dataset
    logger.info('Creating dataset...')
    E3_loader = create_embedding_loaders(E3_seq_dict)
    Sub_loader = create_embedding_loaders(Sub_seq_dict)

    # Create model
    logger.info('Creating model...')
    model = create_embedding_model(device)

    # Perform embedding
    logger.info('Starting embedding...')
    # Note: The Trainer class constructor expects the embedding type as an argument.
    trainer = Trainer(model, device, result_dir, embedding_type='E3')
    E3_embed = trainer.embedding(E3_loader)
    trainer = Trainer(model, device, result_dir, embedding_type='Sub')
    Sub_embed = trainer.embedding(Sub_loader)
    return E3_embed, Sub_embed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings from sequence dictionary files')
    parser.add_argument('--E3_seq_file', type=str, required=True,
                        help='Path to the FASTA file containing E3 sequences')
    parser.add_argument('--Sub_seq_file', type=str, required=True,
                        help='Path to the FASTA file containing Sub sequences')
    args = parser.parse_args()
    E3_fasta = fasta_to_dict(args.E3_seq_file)
    Sub_fasta = fasta_to_dict(args.Sub_seq_file)
    E3_embed, Sub_embed = Embedding(E3_fasta, Sub_fasta)
    print("Successfully generated embeddings.")
