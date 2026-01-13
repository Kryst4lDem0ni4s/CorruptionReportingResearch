"""
Text Utils - Stylometric analysis and text processing

Provides:
- Type-Token Ratio (TTR) calculation
- Lexical density analysis
- Text similarity measures
- Basic POS analysis (without heavy dependencies)
- N-gram extraction
- Text preprocessing
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)


class TextUtils:
    """
    Text utilities for stylometric analysis.
    
    Features:
    - Type-Token Ratio (TTR)
    - Lexical density
    - Text similarity (cosine, Jaccard)
    - N-gram extraction
    - Basic POS detection (simple heuristics)
    - Readability metrics
    """
    
    # Common English function words (for lexical density)
    FUNCTION_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
        'take', 'into', 'year', 'your', 'some', 'could', 'them', 'see', 'other',
        'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think',
        'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
        'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
        'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had', 'were',
        'am', 'being', 'did', 'does', 'done', 'doing'
    }
    
    @staticmethod
    def tokenize(text: str, lowercase: bool = True) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            
        Returns:
            List of tokens
        """
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        
        if lowercase:
            text = text.lower()
        
        tokens = text.split()
        tokens = [t.strip() for t in tokens if t.strip()]
        
        return tokens
    
    @staticmethod
    def calculate_ttr(text: str) -> float:
        """
        Calculate Type-Token Ratio (TTR).
        
        TTR = unique_words / total_words
        Higher TTR indicates more diverse vocabulary.
        
        Args:
            text: Input text
            
        Returns:
            float: TTR value (0-1)
        """
        tokens = TextUtils.tokenize(text)
        
        if not tokens:
            return 0.0
        
        types = set(tokens)
        ttr = len(types) / len(tokens)
        
        logger.debug(f"TTR: {ttr:.3f} (types={len(types)}, tokens={len(tokens)})")
        
        return ttr
    
    @staticmethod
    def calculate_lexical_density(text: str) -> float:
        """
        Calculate lexical density.
        
        Lexical density = content_words / total_words
        Higher density indicates more informational content.
        
        Args:
            text: Input text
            
        Returns:
            float: Lexical density (0-1)
        """
        tokens = TextUtils.tokenize(text)
        
        if not tokens:
            return 0.0
        
        # Count content words (not function words)
        content_words = [t for t in tokens if t not in TextUtils.FUNCTION_WORDS]
        
        density = len(content_words) / len(tokens)
        
        logger.debug(
            f"Lexical density: {density:.3f} "
            f"(content={len(content_words)}, total={len(tokens)})"
        )
        
        return density
    
    @staticmethod
    def extract_ngrams(
        text: str,
        n: int = 2,
        char_level: bool = False
    ) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text
            n: N-gram size
            char_level: Character-level vs word-level
            
        Returns:
            List of n-grams
        """
        if char_level:
            # Character n-grams
            text = text.lower().replace(' ', '')
            ngrams = [
                tuple(text[i:i+n])
                for i in range(len(text) - n + 1)
            ]
        else:
            # Word n-grams
            tokens = TextUtils.tokenize(text)
            ngrams = [
                tuple(tokens[i:i+n])
                for i in range(len(tokens) - n + 1)
            ]
        
        return ngrams
    
    @staticmethod
    def calculate_ngram_frequency(
        text: str,
        n: int = 2
    ) -> Dict[Tuple[str, ...], int]:
        """
        Calculate n-gram frequency distribution.
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            dict: N-gram → frequency
        """
        ngrams = TextUtils.extract_ngrams(text, n=n)
        freq = Counter(ngrams)
        
        return dict(freq)
    
    @staticmethod
    def cosine_similarity(text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Cosine similarity (0-1)
        """
        # Tokenize
        tokens1 = set(TextUtils.tokenize(text1))
        tokens2 = set(TextUtils.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate frequency vectors
        all_tokens = tokens1 | tokens2
        
        vec1 = np.array([1 if t in tokens1 else 0 for t in all_tokens])
        vec2 = np.array([1 if t in tokens2 else 0 for t in all_tokens])
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Jaccard similarity (0-1)
        """
        tokens1 = set(TextUtils.tokenize(text1))
        tokens2 = set(TextUtils.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        return similarity
    
    @staticmethod
    def detect_pos_pattern(text: str) -> Dict[str, int]:
        """
        Detect basic POS patterns using heuristics.
        
        Note: This is a simplified approach without NLTK/spaCy.
        For production, consider using proper POS taggers.
        
        Args:
            text: Input text
            
        Returns:
            dict: POS tag → count
        """
        tokens = TextUtils.tokenize(text, lowercase=False)
        
        pos_counts = defaultdict(int)
        
        for token in tokens:
            # Simple heuristics
            if token.lower() in TextUtils.FUNCTION_WORDS:
                pos_counts['FUNCTION'] += 1
            elif token.isupper():
                pos_counts['PROPER_NOUN'] += 1
            elif token[0].isupper():
                pos_counts['CAPITALIZED'] += 1
            elif token.endswith('ing'):
                pos_counts['VERB_ING'] += 1
            elif token.endswith('ed'):
                pos_counts['VERB_ED'] += 1
            elif token.endswith('ly'):
                pos_counts['ADVERB'] += 1
            else:
                pos_counts['OTHER'] += 1
        
        return dict(pos_counts)
    
    @staticmethod
    def calculate_avg_word_length(text: str) -> float:
        """
        Calculate average word length.
        
        Args:
            text: Input text
            
        Returns:
            float: Average word length in characters
        """
        tokens = TextUtils.tokenize(text)
        
        if not tokens:
            return 0.0
        
        avg_length = sum(len(t) for t in tokens) / len(tokens)
        
        return avg_length
    
    @staticmethod
    def calculate_avg_sentence_length(text: str) -> float:
        """
        Calculate average sentence length.
        
        Args:
            text: Input text
            
        Returns:
            float: Average sentence length in words
        """
        # Split by sentence delimiters
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Count words per sentence
        word_counts = [len(TextUtils.tokenize(s)) for s in sentences]
        avg_length = sum(word_counts) / len(word_counts)
        
        return avg_length
    
    @staticmethod
    def extract_stylometric_features(text: str) -> Dict:
        """
        Extract comprehensive stylometric features.
        
        Args:
            text: Input text
            
        Returns:
            dict: Stylometric feature vector
        """
        tokens = TextUtils.tokenize(text)
        
        features = {
            # Basic statistics
            'num_chars': len(text),
            'num_tokens': len(tokens),
            'num_unique_tokens': len(set(tokens)),
            
            # Lexical features
            'type_token_ratio': TextUtils.calculate_ttr(text),
            'lexical_density': TextUtils.calculate_lexical_density(text),
            'avg_word_length': TextUtils.calculate_avg_word_length(text),
            'avg_sentence_length': TextUtils.calculate_avg_sentence_length(text),
            
            # Character features
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / max(len(text), 1),
        }
        
        # POS pattern
        pos_counts = TextUtils.detect_pos_pattern(text)
        features['pos_pattern'] = pos_counts
        
        # N-gram diversity (2-grams)
        bigrams = TextUtils.extract_ngrams(text, n=2)
        features['bigram_diversity'] = len(set(bigrams)) / max(len(bigrams), 1)
        
        logger.debug(f"Extracted {len(features)} stylometric features")
        
        return features
    
    @staticmethod
    def compare_stylometry(text1: str, text2: str) -> Dict:
        """
        Compare stylometric features between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            dict: Similarity metrics
        """
        features1 = TextUtils.extract_stylometric_features(text1)
        features2 = TextUtils.extract_stylometric_features(text2)
        
        # Calculate feature differences
        numerical_features = [
            'type_token_ratio', 'lexical_density', 'avg_word_length',
            'avg_sentence_length', 'uppercase_ratio', 'digit_ratio',
            'punctuation_ratio', 'bigram_diversity'
        ]
        
        differences = {}
        for feature in numerical_features:
            val1 = features1.get(feature, 0)
            val2 = features2.get(feature, 0)
            differences[f'{feature}_diff'] = abs(val1 - val2)
        
        # Text similarity
        cosine_sim = TextUtils.cosine_similarity(text1, text2)
        jaccard_sim = TextUtils.jaccard_similarity(text1, text2)
        
        comparison = {
            'cosine_similarity': cosine_sim,
            'jaccard_similarity': jaccard_sim,
            'feature_differences': differences,
            'avg_feature_diff': np.mean(list(differences.values()))
        }
        
        return comparison
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        
        # Trim
        text = text.strip()
        
        return text


# Convenience functions

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate average similarity between two texts."""
    cosine = TextUtils.cosine_similarity(text1, text2)
    jaccard = TextUtils.jaccard_similarity(text1, text2)
    return (cosine + jaccard) / 2


def extract_features(text: str) -> Dict:
    """Extract stylometric features from text."""
    return TextUtils.extract_stylometric_features(text)
