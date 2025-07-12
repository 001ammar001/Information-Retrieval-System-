import os
import joblib
import gensim.downloader as api

def train_glove_wiki_gigaword():
    print("processing: ")
    # Save model and vectors
    output_dir = os.path.join("app", "joblib_files")
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, "glove_wiki_gigaword_vectors.joblib")
    print(f"Saving glove_wiki_gigaword model and vectors to '{file_path}'...")
    model = api.load("glove-wiki-gigaword-100")
    joblib.dump(model, file_path)
    print("✅ Successfully saved glove_wiki_gigaword model and vectors.")

if __name__ == "__main__":
    print(f"\nTraining glove wiki gigaword model") 
    train_glove_wiki_gigaword()