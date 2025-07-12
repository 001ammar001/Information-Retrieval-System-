import joblib
import os
from sklearn.decomposition import LatentDirichletAllocation

class TopicDetection:
    def __init__(self):
        self.doc_vectors = {}          # Existing document vectors
        self.lda_models = {}           # New: LDA models per dataset
        self.doc_topics = {}           # New: Topic distributions per document
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
                        self.doc_vectors[dataset_name] = loaded_data['doc_vectors']

                except Exception as e:
                    print(f"Error loading TF-IDF models for {dataset_name}: {e}")
    
    def index_dataset(self, dataset_name):
        """Extend indexing to include topic modeling."""
        # New: Train LDA and store topic distributions
        lda = LatentDirichletAllocation(n_components=75, random_state=42)  # Adjust n_components
        doc_topics = lda.fit_transform(self.doc_vectors[dataset_name])
        self.lda_models[dataset_name] = lda
        self.doc_topics[dataset_name] = doc_topics
        
        output_dir = os.path.join("app", "joblib_files",dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        data_to_save = {
            'lda_models': lda,
            'doc_topics': doc_topics
        }
        file_path = os.path.join(output_dir, "topic_detection_vectors.joblib")
        print(f"Saving topic detection to '{file_path}'...")
        joblib.dump(data_to_save, file_path)
        print("✅ Successfully saved topic detection model and vectors.")
        
topicDetection = TopicDetection()
            

if __name__ == "__main__":
    datasets = ["antique","quora"]
    for dataset_name in datasets:
        print(f"\nTraining Topic detection model for {dataset_name} dataset...")
        topicDetection.index_dataset(dataset_name) 


# topics 5 70% tfidf 30% topic 
# ======================================== RESULTS ========================================
# MAP: 76.85%
# MRR: 82.14%
# Mean Precision: 10.62%
# Mean Recall: 79.46%
# Precision@10: 10.60%
# ==================================================

# # topics 75  70% tfidf 30% topic 
# ======================================== RESULTS ========================================
# MAP: 76.98%
# MRR: 82.34%
# Mean Precision: 10.40%
# Mean Recall: 77.76%
# Precision@10: 10.38%
# ==================================================

# # topics 75  55% tfidf 45% topic 
# ======================================== RESULTS ========================================
# MAP: 76.85%
# MRR: 82.48%
# Mean Precision: 10.05%
# Mean Recall: 75.15%
# Precision@10: 10.03%
# ==================================================


# # topics 125  70% tfidf 30% topic 
# ======================================== RESULTS ========================================
# MAP: 76.91%
# MRR: 82.27%
# Mean Precision: 10.45%
# Mean Recall: 78.06%
# Precision@10: 10.43%
# ==================================================

# # topics 125  55% tfidf 45% topic 
# ======================================== RESULTS ========================================
# MAP: 76.95%
# MRR: 82.59%
# Mean Precision: 10.04%
# Mean Recall: 75.21%
# Precision@10: 10.03%
# ==================================================

# # topics 250  70% tfidf 30% topic 
# ======================================== RESULTS ========================================
# MAP: 75.40%
# MRR: 80.82%
# Mean Precision: 10.30%
# Mean Recall: 77.92%
# Precision@10: 10.29%
# ==================================================

# # topics 250  55% tfidf 45% topic 
# ======================================== RESULTS ========================================
# MAP: 75.65%
# MRR: 81.76%
# Mean Precision: 9.12%
# Mean Recall: 69.67%
# Precision@10: 9.10%
# ==================================================


# # topics 75  70% tfidf 30% topic (antique)
# ======================================== RESULTS ========================================
# MAP: 9.07%
# MRR: 77.07%
# Mean Precision: 34.71%
# Mean Recall: 11.01%
# Precision@10: 33.58%
# ==================================================