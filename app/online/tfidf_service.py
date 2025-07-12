import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from app.online.inverted_index_service import inverted_index_service
import numpy as np

class TFIDFService:
    def __init__(self):
        self.vectorizers = {}
        self.doc_vectors = {}
        self.doc_ids = {}
        self.__load_models()

    def __load_models(self):
        """Load all saved TF-IDF models and document vectors"""
        base_dir = os.path.join("app", "joblib_files")
        for dataset_name in os.listdir(base_dir):
            dataset_path = os.path.join(base_dir, dataset_name)
            if os.path.isdir(dataset_path):
                try:
                    vectors_path = os.path.join(dataset_path, "tf_idf_vectors.joblib")
                    
                    if os.path.exists(vectors_path):
                        loaded_data = joblib.load(vectors_path)
                        self.vectorizers[dataset_name] = loaded_data['fitted_vectorizer']
                        self.doc_vectors[dataset_name] = loaded_data['doc_vectors']
                        self.doc_ids[dataset_name] = loaded_data['doc_ids']

                except Exception as e:
                    print(f"Error loading TF-IDF models for {dataset_name}: {e}")

    def get_top_indices_for_query(self, query, processed_query, dataset_name, top_n):

        matched_documents = inverted_index_service.get_documents_matching_query(processed_query, dataset_name)
        if not matched_documents: return None, None, None
        doc_indices_array = np.array(matched_documents)
        dataset_vectors = self.doc_vectors[dataset_name]
        matched_vector = dataset_vectors[doc_indices_array]

        query_vector = self.vectorizers[dataset_name].transform([query])
        tfidf_scores = cosine_similarity(query_vector, matched_vector).flatten()
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        return top_indices, matched_documents, tfidf_scores
    
    def find_similar_documents(self, query: str, processed_query: str, dataset_name: str, top_n: int = 10):
        if not processed_query or dataset_name not in self.vectorizers:
            return []
        top_indices, matched_documents, tfidf_scores = self.get_top_indices_for_query(query, processed_query, dataset_name, top_n)
        if not matched_documents: return []
        results = []
        for rank, index in enumerate(top_indices, 1):
            doc_id = self.doc_ids[dataset_name][matched_documents[index]]
            score = float(tfidf_scores[index])
            if score == 0: continue
            results.append({
                "rank": rank,
                "document_id": str(doc_id),
                "similarity_score": score,
            })
        return results

# Create a singleton instance
tfidf_service = TFIDFService()
# quora no index
# ======================================== RESULTS ========================================
# MAP: 77.01%
# MRR: 82.07%
# Mean Precision: 10.96%
# Mean Recall: 81.98%
# Precision@10: 10.96%
# ==================================================

# quora index
# ======================================== RESULTS ========================================
# MAP: 77.03%
# MRR: 82.11%
# Mean Precision: 10.96%
# Mean Recall: 81.97%
# Precision@10: 10.96%
# ==================================================


# antique  
# ======================================== RESULTS ========================================
# MAP: 10.83%
# MRR: 85.43%
# Mean Precision: 37.03%
# Mean Recall: 12.60%
# Precision@10: 37.03%
# ==================================================