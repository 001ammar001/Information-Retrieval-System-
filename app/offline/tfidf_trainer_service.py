import os
import joblib
from app.db.db_service import get_db_connection
from app.services.tf_idf_vectorizer import CustomTfIdfVectorizer

def train_tfidf_model(dataset_name: str):
    """
    Train a TF-IDF model on the documents in the specified dataset
    
    Args:
        dataset_name (str): Name of the dataset to train on
    """
    # Get documents from database
    conn = get_db_connection()
    if conn is None:
        print("Database connection could not be established.")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id, content FROM Documents WHERE dataset_name = %s ORDER BY id ASC ", (dataset_name,))
            documents = cur.fetchall()
    finally:
        conn.close()

    if not documents:
        print(f"No documents found for dataset: {dataset_name}")
        return

    print("processing: ", len(documents), "from dataset", dataset_name)

    # Extract document IDs and content
    doc_ids = [doc[0] for doc in documents]
    contents = [doc[1] for doc in documents]

    # Create and fit TF-IDF vectorizer
    custom_vectorizer = CustomTfIdfVectorizer()

    # Transform documents to TF-IDF vectors
    doc_vectors, fitted_vectorizer = custom_vectorizer.calculate_tf_idf_matrix(contents)

    # Save model and vectors
    output_dir = os.path.join("app", "joblib_files",dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    data_to_save = {
        'fitted_vectorizer': fitted_vectorizer,
        'doc_vectors': doc_vectors,
        'doc_ids': doc_ids
    }
    
    file_path = os.path.join(output_dir, "tf_idf_vectors.joblib")
    print(f"Saving TF-IDF model and vectors to '{file_path}'...")
    joblib.dump(data_to_save, file_path)
    print("✅ Successfully saved TF-IDF model and vectors.")

if __name__ == "__main__":
    datasets = ["antique", "quora"]
    for dataset_name in datasets:
        print(f"\nTraining TF-IDF model for {dataset_name} dataset...")
        train_tfidf_model(dataset_name) 