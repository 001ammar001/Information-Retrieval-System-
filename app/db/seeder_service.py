import ir_datasets
from typing import List, Tuple
from .db_service import get_db_connection, insert_documents
from .db_service import create_documents_table, empty_documents_table

def is_database_empty() -> bool:
    """
    Checks if the Documents table is empty.
    
    Returns:
        bool: True if the table is empty, False otherwise
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("Database connection could not be established.")
            return True
        
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM Documents")
            count = cur.fetchone()[0]
            return count == 0
            
    except Exception as error:
        print(f"Error checking database: {error}")
        return True
    finally:
        if conn is not None:
            conn.close()

def load_antique_dataset() -> List[Tuple[str, str, str]]:
    """
    Loads documents from the Antique dataset using ir_datasets.
    
    Returns:
        List[Tuple[str, str, str]]: List of (doc_id, content, dataset_name) tuples
    """
    documents = []
    dataset = ir_datasets.load("antique/test/non-offensive")
    dataset_name = "antique"
    
    print("Processing Antique documents...")
    for doc in dataset.docs_iter():
        # Convert doc_id to string and process the text
        doc_id = str(doc.doc_id)
        content = doc.text
        documents.append((doc_id, content, dataset_name))
    
    return documents

def load_quora_dataset() -> List[Tuple[str, str, str]]:
    """
    Loads documents from the Quora dataset using ir_datasets.
    Downloads the dataset if it doesn't exist.
    
    Returns:
        List[Tuple[str, str, str]]: List of (doc_id, content, dataset_name) tuples
    """
    try:
        # Download the dataset if it doesn't exist
        print("Downloading Quora dataset (this might take a few minutes)...")
        ir_datasets.load("beir/quora/test")
        print("✅ Quora dataset downloaded successfully")
    except Exception as e:
        print(f"Error downloading Quora dataset: {e}")
        return []

    documents = []
    dataset = ir_datasets.load("beir/quora/test")
    dataset_name = "quora"
    
    print("Processing Quora documents...")
    for doc in dataset.docs_iter():
        # Convert doc_id to string and process the text
        doc_id = str(doc.doc_id)
        content = doc.text
        documents.append((doc_id, content, dataset_name))
    
    return documents

def seed_database():
    """
    Seeds the database with documents from both Antique and Quora datasets if the database is empty.
    """
    # Create the documents table first
    print("Creating Documents table...")
    create_documents_table()
    empty_documents_table()
    
    if not is_database_empty():
        print("Database is not empty. Skipping seeding.")
        return

    try:
        # Load and insert Antique dataset
        print("\nLoading Antique dataset...")
        antique_documents = load_antique_dataset()
        if antique_documents:
            print(f"Found {len(antique_documents)} Antique documents to insert.")
            insert_documents(antique_documents)
            print("✅ Antique dataset seeded successfully!")
        else:
            print("No Antique documents found to insert.")

        # Load and insert Quora dataset
        print("\nLoading Quora dataset...")
        qoura_documents = load_quora_dataset()
        if qoura_documents:
            print(f"Found {len(qoura_documents)} Quora documents to insert.")
            insert_documents(qoura_documents)
            print("✅ Quora dataset seeded successfully!")
        else:
            print("No Quora documents found to insert.")

        print("\n✅ Database seeding completed successfully!")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")

if __name__ == "__main__":
    seed_database() 