import os
import joblib
from gensim.models import KeyedVectors

class QuerySuggestion:
    def __init__(self):
        self.model = None
        self.__load_models()
        
    def __load_models(self):
        base_dir = os.path.join("app", "joblib_files")
        if os.path.isdir(base_dir):
            try:
                joblib_path = os.path.join(base_dir, "glove_wiki_gigaword_vectors.joblib")
                if os.path.exists(joblib_path):
                    loaded_data = joblib.load(joblib_path)
                    self.model=loaded_data
                    if not isinstance(self.model, KeyedVectors):
                        raise ValueError("Loaded object is not a Gensim KeyedVectors model")
                    print(f"✅ glove_wiki_gigaword model loaded successfully")
                else:
                    print("⚠️ Model file not found at:", joblib_path)                
            except Exception as e:
                    print(f"Error loading glove_wiki_gigaword models: {e}")   
                    
    def suggest_phrase_completions(self, phrase, top_n=5):
        if not self.model:
            return ["Model not loaded - cannot make suggestions"]
        
        words = phrase.lower().split()
        if not words:
            return []
        
        last_word = words[-1]
        try:
            # Find words that often appear after the last word
            completions = self.model.most_similar(positive=[last_word], topn=top_n)
            
            # Handle cases where the phrase ends with a space
            if phrase.endswith(' '):
                # User is explicitly looking for next word suggestions
                return [f"{phrase}{word}" for word, _ in completions]
            else:
                # User might be mid-word or wanting next word suggestions
                prefix = ' '.join(words[:-1]) + ' ' if len(words) > 1 else ''
                return [f"{prefix}{word}" for word, _ in completions]
        except KeyError:
            return []
        
    def suggest_word_completions(self, partial_word, top_n=5):
        if not self.model:
            return ["Model not loaded - cannot make suggestions"]
        
        if not partial_word:
            return []
        
        partial_word = partial_word.lower()
        try:
            # Get all words in the vocabulary that start with the partial word
            vocab_words = [word for word in self.model.key_to_index.keys() 
                          if word.startswith(partial_word)]
            
            # Sort by frequency (approximated by the word's count in the training data)
            vocab_words.sort(key=lambda word: self.model.get_vecattr(word, 'count'), reverse=True)
            
            # Return top_n suggestions
            return vocab_words[:top_n]
        except KeyError:
            return []
        
    def suggest(self, text, top_n=5):
        """Combined suggestion method that returns both phrase completions and word completions"""
        word_completions = self.suggest_word_completions(text, top_n)
        phrase_completions = self.suggest_phrase_completions(text, top_n)
        print("word_completions",word_completions)
        print("phrase_completions",phrase_completions)
        # Combine and deduplicate suggestions
        combined = []
        seen = set()
        
        # Add word completions first (these are exact matches for the current word)
        for word in word_completions:
            if word not in seen:
                combined.append(word)
                seen.add(word)
        
        # Add phrase completions that aren't already in the list
        for phrase in phrase_completions:
            if phrase not in seen:
                combined.append(phrase)
                seen.add(phrase)
        
        return combined[:top_n]
                    
querySuggestion = QuerySuggestion()