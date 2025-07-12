import os
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from app.services.text_processing_services import text_processing_service
import numpy as np

class BERTService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.doc_vectors = {}
        self.doc_ids = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__load_vectors_and_model()

    def __load_vectors_and_model(self):
        """Load all saved BERT document vectors and model/tokenizer if available"""
        base_dir = os.path.join("app", "joblib_files")
        for dataset_name in os.listdir(base_dir):
            dataset_path = os.path.join(base_dir, dataset_name)
            if os.path.isdir(dataset_path):
                try:
                    vectors_path = os.path.join(dataset_path, "bert_vectors.joblib")
                    if os.path.exists(vectors_path):
                        loaded_data = joblib.load(vectors_path)
                        self.doc_vectors[dataset_name] = loaded_data['doc_vectors']
                        self.doc_ids[dataset_name] = loaded_data['doc_ids']
                        # Load model and tokenizer from local path if available
                        if self.model is None and 'model_dir' in loaded_data and 'tokenizer_dir' in loaded_data:
                            self.tokenizer = AutoTokenizer.from_pretrained(loaded_data['tokenizer_dir'])
                            self.model = AutoModel.from_pretrained(loaded_data['model_dir'])
                            self.model = self.model.to(self.device)
                            print(f"✅ BERT model loaded from local files for {dataset_name} on {self.device}")
                except Exception as e:
                    print(f"Error loading BERT vectors/model for {dataset_name}: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for a text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()

    def find_similar_documents(self, processed_query: str, dataset_name: str, top_n: int = 5):
        if not processed_query or dataset_name not in self.doc_vectors: return []
        query_vector = self.get_embedding(processed_query)
        cosine_similarities = cosine_similarity(query_vector, self.doc_vectors[dataset_name]).flatten()
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
        results = []
        for rank, index in enumerate(top_indices, 1):
            doc_id = self.doc_ids[dataset_name][index]
            score = float(cosine_similarities[index])
            if score == 0: continue
            results.append({
                "rank": rank,
                "document_id": str(doc_id),
                "similarity_score": score
            })
        return results

# Create a singleton instance
bert_service = BERTService()