"""
Tweets Preprocessing Module
Implements Section 3.2 of the TSDA-BERT base paper:
- Lower casing
- URL removal
- User mentions (@user) removal
- Emoticon decoding / polarity conversion
- Hashtag token cleaning
- Punctuation & number removal
- Stopword filtering
"""

import re
from typing import List, Optional

# Basic english stopwords list (self-contained, no external NLTK download required)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't"
}

EMOTICON_DICT = {
    ':)': 'happy',
    ':-)': 'happy',
    ':D': 'laughing',
    ':-D': 'laughing',
    ';)': 'wink',
    ';-)': 'wink',
    ':(': 'sad',
    ':-(': 'sad',
    ':P': 'playful',
    ':-P': 'playful',
    '<3': 'love',
    ':/': 'skeptical',
    ':-/': 'skeptical',
    ':|': 'neutral',
    ':-|': 'neutral',
    ':O': 'surprised',
    ':-O': 'surprised',
    '🔥': 'fire excellent',
    '🚀': 'rocket soaring',
    '📉': 'crash plummet',
    '📈': 'growth increase',
    '💔': 'heartbreak loss',
    '🏆': 'champions trophy',
    '👏': 'applause cheer',
    '😡': 'angry furious',
    '😭': 'crying devastated',
    '🎉': 'celebration joy',
}


class TweetPreprocessor:
    """
    Standard Tweet Preprocessor corresponding to Section 3.2 of Susi & Shanthi (2023).
    """

    def __init__(self, remove_stopwords: bool = True, remove_numbers: bool = True):
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers

    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase
        cleaned = text.lower()

        # 2. Decode Emoticons and common emojis
        for emo, replacement in EMOTICON_DICT.items():
            cleaned = cleaned.replace(emo.lower(), f" {replacement} ")

        # 3. Remove URLs
        cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)

        # 4. Remove User Mentions (@user)
        cleaned = re.sub(r'@\w+', ' ', cleaned)

        # 5. Clean Hashtags (preserve token without '#')
        cleaned = re.sub(r'#(\w+)', r'\1', cleaned)

        # 6. Remove numbers if configured
        if self.remove_numbers:
            cleaned = re.sub(r'\d+', ' ', cleaned)

        # 7. Remove punctuations and special non-alphanumeric chars
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)

        # 8. Tokenize and filter stopwords & whitespace
        tokens = cleaned.split()
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
        else:
            tokens = [w for w in tokens if len(w) > 1]

        return ' '.join(tokens)

    def extract_word_set(self, text: str) -> set:
        """Returns set of cleaned words W in tweet for Domain Impact Score calculation."""
        cleaned = self.clean_text(text)
        return set(cleaned.split())


# Global singleton instance for quick reuse
preprocessor = TweetPreprocessor()
