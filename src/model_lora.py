import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig, TaskType

def setup_esm_model():
    # Load the pre-trained model
    model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True)
    tokenizer = model.tokenizer
    
    return model, tokenizer

def add_lora_layers(base_model):
    """
    Add LORA layers to the base model.
    INPUT:
        base_model: The base model to which LORA layers will be added.
    OUTPUT:
        model: The model with LORA layers added.
    """
    
    # Add LORA layers to the base model
    # LORA configuration
    target_modules = [
        "10.attn.layernorm_qkv.1", "11.attn.layernorm_qkv.1", "12.attn.layernorm_qkv.1", 
        "13.attn.layernorm_qkv.1", "14.attn.layernorm_qkv.1", "15.attn.layernorm_qkv.1", 
        "16.attn.layernorm_qkv.1", "17.attn.layernorm_qkv.1", "18.attn.layernorm_qkv.1", 
        "19.attn.layernorm_qkv.1", "20.attn.layernorm_qkv.1", "21.attn.layernorm_qkv.1", 
        "22.attn.layernorm_qkv.1", "23.attn.layernorm_qkv.1", "24.attn.layernorm_qkv.1", 
        "25.attn.layernorm_qkv.1", "26.attn.layernorm_qkv.1", "27.attn.layernorm_qkv.1", 
        "28.attn.layernorm_qkv.1", "29.attn.layernorm_qkv.1"
    ]

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=4,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
    )

    model_with_lora = get_peft_model(base_model, lora_config)  
    return model_with_lora


class EmbeddingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_embedding_loaders(sequences, labels, batch_size=32, shuffle=True):
    dataset = EmbeddingDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class EmbeddingModel(nn.Module):
    def __init__(self, model_with_lora, tokenizer):
        super(EmbeddingModel, self).__init__()
        """
        Initialize the ESMcLoRA model.
        INPUT:
            model_with_lora: The ESM model with LORA layers added.
            tokenizer: The tokenizer for the ESM model.
        """
        
        self.model = model_with_lora
        self.tokenizer = tokenizer

    def preprocess(self, seqs):
        """
        Process the list of sequences.
        """
        # Get the length of each sequence
        seq_len = [len(seq) for seq in seqs]
        # Tokenize the sequences
        tokenized = self.tokenizer(seqs, return_tensors='pt', padding=True)
        return seq_len, tokenized.to(self.model.device)
    
    def postprocess(self, output, seq_len):
        """
        Post-process the output of the ESM model.
        """
        # Get the hidden states from the last layer
        last_hidden = output.last_hidden_state

        # Perform mean pooling (excluding first and last token)
        pooled = [emb[1:l-1, :].mean(dim=0) for emb, l in zip(last_hidden, seq_len)]
        pooled = torch.stack(pooled, dim=0)
        return pooled
    
    def forward(self, seq):
        # Preprocess the sequences
        seq_len, seq_input = self.preprocess(seq)
        # Perform ESM embedding
        output = self.model(**seq_input)
        # Post-process the output
        output = self.postprocess(output, seq_len)
      
        return output

