import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.online.bert_service import bert_service
from app.online.tfidf_service import tfidf_service

class HybridService:
    def _rerank_using_bert_embeddings(self, processed_query, doc_indices, dataset_name):
        query_embedding = bert_service.get_embedding(processed_query)
        
        doc_embeddings = np.array([bert_service.doc_vectors[dataset_name][i] for i in doc_indices])
       
        if query_embedding.ndim > 1: query_embedding = query_embedding.squeeze(0)
        
        bert_scores = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
        return bert_scores

    def find_similar_documents(self, query: str, processed_query: str, dataset_name: str, top_n: int = 10):
        top_indices, matched_documents, tfidf_scores = tfidf_service.get_top_indices_for_query(query, processed_query, dataset_name, top_n * 20)
        if not matched_documents: return []
        matched_documents = np.array(matched_documents)
        candidate_indices = matched_documents[top_indices]
        try:
            bert_scores = self._rerank_using_bert_embeddings(processed_query, candidate_indices, dataset_name)
            combined_scores = 0.6 * tfidf_scores[top_indices] + 0.4 * bert_scores
            score_type = "hybrid"
        except Exception as e:
            print(f"Using TF-IDF only due to BERT error: {str(e)}")
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
    
hybrid_service = HybridService() 