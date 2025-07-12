from app.db.db_service import get_documents_by_id
from app.online.tfidf_service import tfidf_service
from app.online.bert_service import bert_service
from app.online.hybrid_service  import hybrid_service
from app.online.topic_detection_service import topic_detection_service
import requests

class SearchService:
    def search(self, query: str, dataset_name: str, top_n: int = 5, method: str = "tfidf"):
        if not query: return []
        response = requests.get(f"http://localhost:5000/process_text?query={query}")
        processed_query = response.json()["processed_query"]
        print(processed_query)
        method = method.lower()
        if method == "bert":
            results = bert_service.find_similar_documents(processed_query, dataset_name, top_n)
        elif method == "hybrid":
            results = hybrid_service.find_similar_documents(query, processed_query, dataset_name, top_n)
        elif method == "topicdetection":
            results = topic_detection_service.find_similar_documents(query, processed_query, dataset_name, top_n)
        else:
            results = tfidf_service.find_similar_documents(query, processed_query, dataset_name, top_n)
        doc_ids = [result["document_id"] for result in results]
        documents = get_documents_by_id(doc_ids, dataset_name)
        doc_content_map = {doc[0]: doc[1] for doc in documents}
        for result in results:
            result["content"] = doc_content_map.get(result["document_id"], "Content not found")
        return results

search_service = SearchService()