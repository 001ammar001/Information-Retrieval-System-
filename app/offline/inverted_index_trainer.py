import os
import json
import joblib

class InvertedIndexTranierService:
    def train_inverted_index_from_corpus(self,dataset_name):
        print(f"\nProcessing {dataset_name} dataset...")
        json_path = os.path.join("app", "processed_datasets", f"{dataset_name}.json")
        if not os.path.exists(json_path):
            print(f"❌ Processed JSON file not found for {dataset_name}")
            return
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            inverted_index = {}
            
            for idx, content in enumerate(corpus_data.values()):
                words = content.split()
                for word in set(words):
                    if word not in inverted_index:
                        inverted_index[word] = []
                        
                    inverted_index[word].append(idx)

            output_dir = os.path.join("app", "inverted_index", dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "inverted_index.joblib")
            joblib.dump(inverted_index, file_path)
            print(f"✅ Saved inverted index for {dataset_name} at {file_path}")

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {str(e)}")

inverted_index_trainer_service = InvertedIndexTranierService()

if __name__ == "__main__":
    datasets = ["antique","quora"]
    for dataset_name in datasets:
        print(f"\nGenerating Inverted Index for {dataset_name} dataset...")
        inverted_index_trainer_service.train_inverted_index_from_corpus(dataset_name)