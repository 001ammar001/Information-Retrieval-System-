import os
import joblib

class InvertedIndexService:
    def __init__(self):
        self.inverted_indexs = {}
        self.__load_index()

    def __load_index(self):
        """Load or create inverted index for all datasets"""
        base_dir = os.path.join("app", "inverted_index")
        for dataset_name in os.listdir(base_dir):
            dataset_path = os.path.join(base_dir, dataset_name)
            if os.path.isdir(dataset_path):
                try:
                    inverted_index_path = os.path.join(dataset_path, "inverted_index.joblib")
                    if os.path.exists(inverted_index_path):
                        self.inverted_indexs[dataset_name] = joblib.load(inverted_index_path)
                        print(f"✅ Loaded inverted index for {dataset_name}")
                except Exception as e:
                    print(f"Error loading inverted index for {dataset_name}: {e}")

    def get_documents_matching_query(self, processed_query: str, dataset_name: str):
        candidate_doc_ids = set()
        query_terms = processed_query.split()
        dataset_index = self.inverted_indexs[dataset_name]
        for term in query_terms:
            if term in dataset_index: 
                candidate_doc_ids.update(dataset_index[term])
        return list(candidate_doc_ids)    

inverted_index_service = InvertedIndexService()