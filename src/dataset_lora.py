from torch.utils.data import Dataset, DataLoader

# Function to convert a FASTA file to a .pt file
def convert_fasta_to_pt(fasta_file):
    """
    Function to read a FASTA file and convert it to a .pt file.
    """
    sequences = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_id = line[1:].strip()
            else:
                seq = line.strip()
                sequences[seq_id] = seq
    return sequences

class EmbeddingDataset(Dataset):
    """
    Dataset class for generating embeddings using a pre-trained ESM-c model.
    """
    def __init__(self, sequence_dict, max_length=2046):
        self.sequences = sequence_dict
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Get protein ID
        seq_id = list(self.sequences.keys())[idx]
        seq = self.sequences[seq_id]
        
        # Trim the sequence
        seq = seq[:self.max_length]
        
        return {
            'seq_id': seq_id,
            'seq': seq
        }
        
def create_embedding_loaders(embeddings_dict):
    """
    Create an embedding dataset and return a DataLoader.
    """
    embedding_dataset = EmbeddingDataset(embeddings_dict)
    embedding_dataloader = DataLoader(
        embedding_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    
    return embedding_dataloader