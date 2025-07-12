import os
import joblib
# from app.online.bert_service import bert_service
from app.online.tfidf_service import tfidf_service
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TopicDetection:
    def __init__(self):
        self.vectorizers = {}
        self.doc_vectors = {}
        self.doc_ids = {}
        self.lda_models = {}           
        self.doc_topics = {}          
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
                        print(f"✅ Loaded tf-idf for {dataset_name}")

                    vectors_path=os.path.join(dataset_path, "topic_detection_vectors.joblib")
                    if os.path.exists(vectors_path):
                        loaded_data = joblib.load(vectors_path)
                        self.lda_models[dataset_name]=loaded_data['lda_models']
                        self.doc_topics[dataset_name]=loaded_data['doc_topics']
                        print(f"✅ Loaded topic detection for {dataset_name}")
                        
                except Exception as e:
                    print(f"Error loading TF-IDF and topic detection models for {dataset_name}: {e}")
                    
    def _rerank_using_bert_embeddings(self, processed_query, doc_indices, dataset_name):
        """Get BERT similarity scores for the given documents"""
        try:
            query_embedding = bert_service.get_embedding(processed_query)
            doc_embeddings = np.array([bert_service.doc_vectors[dataset_name][i] for i in doc_indices])
            
            if query_embedding.ndim > 1: 
                query_embedding = query_embedding.squeeze(0)
            
            bert_scores = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
            return bert_scores
        except Exception as e:
            print(f"Error getting BERT scores: {e}")
            return np.zeros(len(doc_indices))

    def _get_topic_scores(self, query, doc_indices, dataset_name):
        """Get topic detection scores for the given documents"""
        try:
            # Get topic vectors for the candidate documents
            dataset_topics = self.doc_topics[dataset_name][doc_indices]
            
            # Transform query to topic space
            query_vector = self.vectorizers[dataset_name].transform([query])
            query_topics = self.lda_models[dataset_name].transform(query_vector)
            
            # Calculate topic similarity scores
            topic_scores = cosine_similarity(query_topics, dataset_topics).flatten()
            return topic_scores
        except Exception as e:
            print(f"Error getting topic scores: {e}")
            return np.zeros(len(doc_indices))
                    
    def find_similar_documents(self, query: str, processed_query: str, dataset_name: str, top_n: int = 5):
        # Use the same approach as hybrid service
        top_indices, matched_documents, tfidf_scores = tfidf_service.get_top_indices_for_query(query, processed_query, dataset_name, top_n * 20)
        if not matched_documents: 
            return []
        
        matched_documents = np.array(matched_documents)
        candidate_indices = matched_documents[top_indices]
        
        try:
            # Get BERT scores
            bert_scores = self._rerank_using_bert_embeddings(processed_query, candidate_indices, dataset_name)
            
            # Get topic detection scores
            topic_scores = self._get_topic_scores(query, candidate_indices, dataset_name)
            
            # Combine all three scores with specified weights
            tfidf_weight = 0.6
            bert_weight = 0.3
            topic_weight = 0.1
            
            combined_scores = (tfidf_weight * tfidf_scores[top_indices] + 
                             bert_weight * bert_scores + 
                             topic_weight * topic_scores)
            score_type = "enhanced_topic_detection"
            
        except Exception as e:
            print(f"Using TF-IDF only due to error: {str(e)}")
            combined_scores = tfidf_scores[top_indices]
            score_type = "tfidf"
        
        results = []
        for rank, score_idx in enumerate(np.argsort(combined_scores)[-top_n:][::-1], 1):
            doc_idx = candidate_indices[score_idx]
            doc_id = tfidf_service.doc_ids[dataset_name][doc_idx]
            results.append({
                "rank": rank,
                "document_id": str(doc_id),
                "similarity_score": float(combined_scores[score_idx]),
                "method": score_type
            })
        return results
    
topic_detection_service = TopicDetection()
