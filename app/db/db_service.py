import psycopg2
import psycopg2.extras
from psycopg2.sql import SQL, Identifier
import os

def create_database_if_not_exists():
    db_name = os.environ.get("DB_NAME", "postgres")


    conn_params = {
        "host": os.environ.get("DB_HOST", "ir-postgres"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASSWORD", "postgres"),
        "port": os.environ.get("DB_PORT", "5436"),
        "dbname": "postgres"
    }

    conn = None
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True  # CREATE DATABASE cannot run inside a transaction block

        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone() is None:
                print(f"Database '{db_name}' does not exist. Creating it now...")
                # Use psycopg2.sql to safely quote the database name
                cur.execute(SQL("CREATE DATABASE {}").format(Identifier(db_name)))
                print(f"Database '{db_name}' created successfully.")
            else:
                print(f"Database '{db_name}' already exists.")
    except psycopg2.Error as e:
        print(f"Error: Could not create database. {e}")
        print("Please ensure the specified user has CREATEDB privileges in PostgreSQL.")
    finally:
        if conn:
            conn.close()

def get_db_connection():
    """
    Establishes a connection to the target PostgreSQL database.
    Assumes the database has already been created.
    """
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            dbname=os.environ.get("DB_NAME", "ir"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "1234awsd"),
            port=os.environ.get("DB_PORT", "5432")


        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to the PostgreSQL database '{os.environ.get('DB_NAME', 'ir')}'. {e}")
        return None

def create_documents_table():
    """
    Creates the 'Documents' table in the database if it does not already exist.
    The table includes an auto-incrementing id field and a unique constraint on (doc_id, dataset_name).
    """
    conn = None
    try:
        create_table_command = """
        CREATE TABLE IF NOT EXISTS Documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(255) NOT NULL,
            content TEXT,
            dataset_name VARCHAR(255) NOT NULL,
            UNIQUE(doc_id, dataset_name)
        );
        """
        conn = get_db_connection()
        if conn is None:
            print("Table creation failed because database connection could not be established.")
            return

        with conn.cursor() as cur:
            cur.execute(create_table_command)
            conn.commit()
            print("Table 'Documents' created successfully or already exists.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while creating PostgreSQL table: {error}")
        if conn:
            conn.rollback()
    finally:
        if conn is not None:
            conn.close()

def insert_documents(documents):
    """
    Inserts or updates documents in the 'Documents' table.
    If a document with the same (doc_id, dataset_name) already exists, it updates the content.
    
    Args:
        documents (list of tuples): A list of documents to insert, where each tuple is
                                     (doc_id, content, dataset_name).
    """
    conn = None
    sql = """
    INSERT INTO Documents (doc_id, content, dataset_name) VALUES %s
    ON CONFLICT (doc_id, dataset_name) DO UPDATE SET content = EXCLUDED.content;
    """
    try:
        conn = get_db_connection()
        if conn is None:
            print("Document insertion failed because database connection could not be established.")
            return
        
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, documents)
            conn.commit()
            print(f"{len(documents)} documents inserted/updated successfully.")
            
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error during document insertion: {error}")
        if conn:
            conn.rollback()
    finally:
        if conn is not None:
            conn.close()

def get_documents_by_id(doc_ids, dataset_name):
    """
    Retrieves documents from the 'Documents' table based on a list of doc_ids and a dataset_name.
    
    Args:
        doc_ids (list of str): A list of document IDs to retrieve.
        dataset_name (str): The name of the dataset to search within.
        
    Returns:
        list of tuples: A list of the found documents, or an empty list if none are found.
    """
    conn = None
    sql = "SELECT doc_id, content, dataset_name FROM Documents WHERE dataset_name = %s AND doc_id = ANY(%s);"
    documents = []
    try:
        conn = get_db_connection()
        if conn is None:
            print("Document retrieval failed because database connection could not be established.")
            return documents
        
        with conn.cursor() as cur:
            cur.execute(sql, (dataset_name, doc_ids))
            documents = cur.fetchall()
            print(f"Found {len(documents)} documents matching the criteria.")
            
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error during document retrieval: {error}")
    finally:
        if conn is not None:
            conn.close()
    return documents

def empty_documents_table():
    conn = None
    sql = "TRUNCATE TABLE Documents CASCADE;"
    try:
        conn = get_db_connection()
        if conn is None:
            print("Table truncation failed because database connection could not be established.")
            return
        
        with conn.cursor() as cur:
            print("Truncating 'Documents' table...")
            cur.execute(sql)
            conn.commit()
            print("'Documents' table has been emptied.")
            
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while truncating table: {error}")
        if conn:
            conn.rollback()
    finally:
        if conn is not None:
            conn.close()