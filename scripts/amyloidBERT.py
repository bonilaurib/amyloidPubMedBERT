import os
import torch
from transformers import BertTokenizer, BertModel
import time
import gc

# Define the device (CUDA, MPS, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
model = BertModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract").to(device)
model.eval()

def load_abstracts(file_path):
    """Load abstracts from the specified text file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            abstracts = content.split('\nAbstract #')
            abstracts = ['Abstract #' + abstract for abstract in abstracts[1:]]
        print(f"Loaded {len(abstracts)} abstracts.")
        return abstracts
    except Exception as e:
        print(f"Error loading abstracts: {e}")
        return []

def tokenize_abstracts(abstracts, batch_size=50):
    """Tokenize abstracts in batches."""
    token_lengths = []  # Debugging token lengths
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i + batch_size]
        tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        token_lengths.extend([len(ids) for ids in tokenized['input_ids']])
        yield tokenized

    print(f"Average token length: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"Maximum token length: {max(token_lengths)}")

def generate_embeddings(tokenized_batch):
    """Generate embeddings from the tokenized batch."""
    input_ids = tokenized_batch['input_ids'].to(device)
    attention_mask = tokenized_batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] embeddings
    
    return embeddings

def save_embeddings(embeddings_list, file_name="embeddings.pt", append=False):
    """Save generated embeddings to a file."""
    all_embeddings = torch.cat(embeddings_list, dim=0)
    print(f"Final embeddings shape: {all_embeddings.shape}")

    if append and os.path.exists(file_name):
        # Load existing embeddings and move to the correct device
        existing = torch.load(file_name).to(device)
        # Concatenate existing and new embeddings
        all_embeddings = torch.cat([existing, all_embeddings], dim=0)
    
    # Save embeddings to the CPU for compatibility
    torch.save(all_embeddings.cpu(), file_name)

def main():
    """Main function to run the embedding generation process."""
    start_time = time.time()

    abstracts = load_abstracts('unique_abstracts.txt')

    # Checkpoint for resuming progress
    embeddings_file = "embeddings.pt"
    processed_count = 0
    if os.path.exists(embeddings_file):
        existing_embeddings = torch.load(embeddings_file)
        processed_count = existing_embeddings.shape[0]
        print(f"{processed_count} abstracts already processed. Resuming...")

    embeddings_list = []
    save_interval = 10  # Save every 10 batches

    try:
        for i, tokenized_batch in enumerate(tokenize_abstracts(abstracts[processed_count:], batch_size=50), start=1):
            embeddings = generate_embeddings(tokenized_batch)
            print(f"Generated embeddings with shape: {embeddings.shape}")
            embeddings_list.append(embeddings)

            if i % save_interval == 0:
                save_embeddings(embeddings_list, embeddings_file, append=True)
                embeddings_list.clear()  # Clear memory
                gc.collect()  # Run garbage collector

        if embeddings_list:  # Save remaining embeddings
            save_embeddings(embeddings_list, embeddings_file, append=True)
    
    except Exception as e:
        print(f"Error during embeddings generation: {e}")
    
    print("Embeddings successfully generated and saved.")
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
