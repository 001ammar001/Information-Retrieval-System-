from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.text_processing_services import text_processing_service

def tokenize(text:str): return text.split(" ")

class CustomTfIdfVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            preprocessor=text_processing_service.process_text,
            lowercase=False,
            token_pattern=None,
            # min_df=0.001,
            # max_df=0.95
            # max_features=10000000
        )
    
    def calculate_tf_idf_matrix(self, docs: list[str]):
        tfidf_matrix = self.vectorizer.fit_transform(docs)
        return tfidf_matrix, self.vectorizer

    def find_cosine_sim(self, tfidf_matrix, query):
        query_vec = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        ranked_indices = cosine_sim.argsort()[::-1]
        return [(i, cosine_sim[i]) for i in ranked_indices] 