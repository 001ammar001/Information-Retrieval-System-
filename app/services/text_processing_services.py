import string
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from typing import List, Tuple
import re

REMOVE_URLS_REGEX = r"(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])"


class TextProcessingService:
    def __init__(self):
        self.spell_checker = SpellChecker(distance=1)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words_set = set(stopwords.words("english"))
        self.TAG_DICT = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

    def __correct_sentence_spelling(self, tokens: List[str]) -> List[str]:
        """Correct spelling in a list of tokens."""
        return [self.spell_checker.correction(token) or token for token in tokens]

    def __get_wordnet_pos(self, tag_parameter: str) -> str:
        """Get WordNet POS tag from Penn Treebank tag."""
        tag = tag_parameter[0].upper()
        return self.TAG_DICT.get(tag, wordnet.NOUN)

    def __lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize a list of tokens using WordNet POS tags."""
        tagged_tokens: List[Tuple[str, str]] = pos_tag(tokens)
        return [
            self.lemmatizer.lemmatize(word, pos=self.__get_wordnet_pos(tag))
            for word, tag in tagged_tokens
        ]
    
    def __remove_punctuation(self, tokens: List[str]):
        table = str.maketrans("", "", string.punctuation)
        return [t.translate(table) for t in tokens if t.translate(table)]


    def process_text(self, text: str) -> str:
        if not text: return ""

        lowered_text = text.lower()

        text_without_urls = re.sub(REMOVE_URLS_REGEX, "", lowered_text)

        tokens = word_tokenize(text_without_urls)
        
        valid_tokens = self.__remove_punctuation(tokens)
        
        valid_tokens = self.__correct_sentence_spelling(valid_tokens)

        valid_tokens = [
            token for token in valid_tokens if token not in self.stop_words_set
        ]

        valid_tokens = self.__lemmatize_tokens(valid_tokens)
        
        return " ".join(valid_tokens)
    
text_processing_service = TextProcessingService()