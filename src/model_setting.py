import torch
from src.model_lora import setup_esm_model, add_lora_layers, EmbeddingModel


def strip_prefix(state_dict, prefix):
    """
    Remove the specified prefix from state_dict keys.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def create_embedding_model(device):
    """
    Create the embedding model.
    This function creates a model by adding LoRA to the ESM-c model.
    """
    # Create the base model and tokenizer
    base_model, tokenizer = setup_esm_model()
    model_with_lora = add_lora_layers(base_model)
    model_with_lora = model_with_lora.to(device)
    
    model = EmbeddingModel(model_with_lora, tokenizer)
    
    # Load only LoRA parameters
    state_dict = torch.load("params/lora_param.pt")
    state_dict = strip_prefix(state_dict, "module.")
    filtered_param  = {k: v for k, v in state_dict.items() if "lora" in k}
    
    # Update the model with the LoRA parameters
    model_dict = model.state_dict()
    model_dict.update(filtered_param)
    model.load_state_dict(model_dict)
    
    return model
