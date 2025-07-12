import ir_datasets
import json
from app.services.text_processing_services import text_processing_service
import os

JSON_FOLDER = os.path.join("app", "processed_datasets")
os.makedirs(JSON_FOLDER, exist_ok=True)

def process_and_save_dataset():
    processed = {}
    for dataset, dataset_name in [
        (ir_datasets.load("antique/test/non-offensive"), "antique"),
        (ir_datasets.load("beir/quora/test"), "quora")
    ]:
        print(f"Loading {dataset_name} dataset...")
        for doc in dataset.docs_iter():
            doc_id = str(doc.doc_id)
            content = doc.text
            processed_text = text_processing_service.process_text(content)
            processed[doc_id] = processed_text
        
        DATASET_FOLDER = os.path.join(JSON_FOLDER, f"{dataset_name}.json")
        
        with open(DATASET_FOLDER, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved processed dataset to {JSON_FOLDER}")

if __name__ == "__main__":
    process_and_save_dataset() 