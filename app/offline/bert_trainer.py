import os
import joblib
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json

def train_bert_model(dataset_name: str):
    print(f"\nProcessing {dataset_name} dataset...")
    json_path = os.path.join("app", "processed_datasets", f"{dataset_name}.json")
    if not os.path.exists(json_path):
        print(f"❌ Processed JSON file not found for {dataset_name}")
        return
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {str(e)}")

    doc_ids = list(corpus_data.keys())
    contents = list(corpus_data.values())

    # Initialize BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Process documents in batches
    batch_size = 4
    doc_vectors = []
    
    print(f"Generating BERT embeddings for {len(contents)} documents...")
    for i in tqdm(range(0, len(contents), batch_size)):
        batch_texts = contents[i:i + batch_size]
        
        # Tokenize and prepare input
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling to get sentence embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            doc_vectors.extend(embeddings.cpu().numpy())

    # Convert to numpy array
    doc_vectors = np.array(doc_vectors)

    # Save vectors, doc_ids, model, and tokenizer
    output_dir = os.path.join("app", "joblib_files", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save model and tokenizer locally
    model_dir = os.path.join(output_dir, "bert_model")
    tokenizer_dir = os.path.join(output_dir, "bert_tokenizer")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    
    data_to_save = {
        'doc_vectors': doc_vectors,
        'doc_ids': doc_ids,
        'model_dir': model_dir,
        'tokenizer_dir': tokenizer_dir
    }
    
    file_path = os.path.join(output_dir, "bert_vectors.joblib")
    print(f"Saving BERT vectors and model to '{file_path}'...")
    joblib.dump(data_to_save, file_path)
    print("✅ Successfully saved BERT vectors and model.")

if __name__ == "__main__":
    # Train models for each dataset
    datasets = ["antique","quora"]  # Add more datasets as needed
    for dataset_name in datasets:
        print(f"\nGenerating BERT embeddings for {dataset_name} dataset...")
        train_bert_model(dataset_name) 

# import os
# import joblib
# import torch
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel
# from app.db.db_service import get_db_connection
# from app.services.text_processing_services import text_processing_service

# #TODO: check if generating embedding in the same time when we procsess will change the results

# def train_bert_model(dataset_name: str):
#     print(f"\nProcessing {dataset_name} dataset...")
    
#     conn = get_db_connection()
#     if conn is None:
#         print("Database connection could not be established.")
#         return

#     try:
#         with conn.cursor() as cur:
#             cur.execute("SELECT doc_id, content FROM Documents WHERE dataset_name = %s ORDER BY id ASC ", (dataset_name,))
#             documents = cur.fetchall()
#     finally:
#         conn.close()

#     if not documents:
#         print(f"No documents found for dataset: {dataset_name}")
#         return

#     print("processing: ", len(documents), "from dataset", dataset_name)

#     # Extract document IDs and content
#     doc_ids = [doc[0] for doc in documents]
#     contents = [doc[1] for doc in documents]

#     # Initialize BERT model and tokenizer
#     print("Loading BERT model and tokenizer...")
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
    
#     # Move model to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     print(f"Model loaded on {device}")

#     # Process documents in batches
#     batch_size = 4
#     doc_vectors = []
#     print(f"Processing the texts")
#     contents = [text_processing_service.process_text(text) for text in contents]
    
#     print(f"Generating BERT embeddings for {len(contents)} documents...")
#     for i in tqdm(range(0, len(contents), batch_size)):
#         batch_texts = contents[i:i + batch_size]
        
#         # Tokenize and prepare input
#         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
        
#         # Get model output
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # Use mean pooling to get sentence embeddings
#             embeddings = outputs.last_hidden_state.mean(dim=1)
#             doc_vectors.extend(embeddings.cpu().numpy())

#     # Convert to numpy array
#     doc_vectors = np.array(doc_vectors)

#     # Save vectors, doc_ids, model, and tokenizer
#     output_dir = os.path.join("app", "joblib_files", dataset_name)
#     os.makedirs(output_dir, exist_ok=True)

#     # Save model and tokenizer locally
#     model_dir = os.path.join(output_dir, "bert_model")
#     tokenizer_dir = os.path.join(output_dir, "bert_tokenizer")
#     model.save_pretrained(model_dir)
#     tokenizer.save_pretrained(tokenizer_dir)
    
#     data_to_save = {
#         'doc_vectors': doc_vectors,
#         'doc_ids': doc_ids,
#         'model_dir': model_dir,
#         'tokenizer_dir': tokenizer_dir
#     }
    
#     file_path = os.path.join(output_dir, "bert_vectors.joblib")
#     print(f"Saving BERT vectors and model to '{file_path}'...")
#     joblib.dump(data_to_save, file_path)
#     print("✅ Successfully saved BERT vectors and model.")

# if __name__ == "__main__":
#     # Train models for each dataset
#     datasets = ["antique","quora"]  # Add more datasets as needed
#     for dataset_name in datasets:
#         print(f"\nGenerating BERT embeddings for {dataset_name} dataset...")
#         train_bert_model(dataset_name) 