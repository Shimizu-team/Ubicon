import torch
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, device, result_dir, embedding_type):
        """
        INPUT:
            model: Model to be used.
            device: Device to be used.
            result_dir: Directory to save results.
            embedding_type: Type of embedding (e.g., 'E3' or 'Sub').
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.output_dir = result_dir
        self.embedding_type = embedding_type
    
    ## MARK: Embedding
    def embedding(self, dataloader):
        """
        Function to perform embedding.
        INPUT:
            dataloader: DataLoader for the dataset.
        """
        self.model.eval()
        os.makedirs(os.path.join(self.output_dir, "embeddings"), exist_ok=True)
        dic = {}
        # Process data batch-by-batch using the dataloader
        with torch.no_grad():
            for batch in dataloader:
                seq_ids = batch['seq_id']
                seqs = batch['seq']
                output = self.model(seqs)

                # Create a dictionary for the current mini-batch
                batch_dic = {seq_id: pool for seq_id, pool in zip(seq_ids, output.cpu())}
                # Add to the main dictionary
                dic.update(batch_dic)
                
            # Save the embeddings to a file
            save_path = os.path.join(self.output_dir, "embeddings", f"{self.embedding_type}_feature_embedding.pt")
            torch.save(dic, save_path)
            logger.info(f"Embeddings saved to {save_path}")

        return dic