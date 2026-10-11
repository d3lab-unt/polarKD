"""
Baseline Methods for Knowledge Graph Evaluation
Implements: Co-occurrence, Rule-based, TF-IDF Only, YAKE Only
"""

import re
import spacy
import yake
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not loaded. Rule-based baseline may not work.")
    nlp = None


class BaselineCooccurrence:
    """
    Baseline 1: Simple Co-occurrence
    Extracts relationships based on keyword co-occurrence in text windows
    """

    def __init__(self, window_size=50):
        self.window_size = window_size

    def extract_keywords(self, text, k=15):
        """Extract top k keywords by frequency (includes n-grams 1-3)"""
        # Use TF-IDF with n-grams to extract multi-word phrases
        from sklearn.feature_extraction.text import TfidfVectorizer

        stop_words = set(stopwords.words('english'))

        vectorizer = TfidfVectorizer(
            max_features=k*2,  # Get more candidates
            ngram_range=(1, 3),  # 1-3 word phrases
            stop_words=list(stop_words),
            token_pattern=r'\b[a-z]{3,}\b'  # At least 3 chars
        )

        try:
            X = vectorizer.fit_transform([text])
            candidates = vectorizer.get_feature_names_out()

            # Rank by TF-IDF scores
            scores = X.toarray()[0]
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

            # Return top k
            keywords = [phrase for phrase, score in ranked[:k]]
            return keywords
        except:
            # Fallback to simple frequency if TF-IDF fails
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            words = [w for w in words if w not in stop_words]
            freq = Counter(words)
            return [word for word, count in freq.most_common(k)]

    def extract_relations(self, text, keywords):
        """Extract co-occurrence relationships"""
        relations = []
        words = text.lower().split()

        keyword_set = set(kw.lower() for kw in keywords)

        for i, word in enumerate(words):
            if word in keyword_set:
                # Look in window
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size)
                window = words[start:end]

                for other_kw in keyword_set:
                    if other_kw in window and other_kw != word:
                        # Avoid duplicates
                        pair = tuple(sorted([word, other_kw]))
                        relations.append({
                            'source': pair[0],
                            'relation': 'CO_OCCURS_WITH',
                            'target': pair[1]
                        })

        # Remove duplicates
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel['source'], rel['relation'], rel['target'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

    def process(self, text, k=15):
        """Main processing pipeline"""
        keywords = self.extract_keywords(text, k)
        relations = self.extract_relations(text, keywords)

        return {
            'nodes': keywords,
            'edges': relations
        }


class BaselineRuleBased:
    """
    Baseline 2: Rule-based extraction using dependency parsing
    Uses spaCy for syntactic analysis
    """

    def __init__(self):
        self.nlp = nlp

    def is_valid_keyword(self, phrase):
        """Filter out junk keywords (journal names, URLs, DOIs, etc.)"""
        junk_patterns = [
            r'doi', r'fig\.', r'geophys', r'res\.', r'lett\.',
            r'http', r'www\.', r'\.org', r'\.com', r'@',
            r'et al', r'etal', r'\.pdf', r'volume',
            r'trans\.', r'proc\.', r'conf\.',
            r'journal', r'science', r'nature',
            r'preprint', r'submitted', r'revision',
            r'copyright', r'license', r'[0-9]{4}',  # Years
            r'^[0-9]+$',  # Pure numbers
            r'^\w{1,2}$',  # Single/two letter words
        ]

        phrase_lower = phrase.lower()

        # Check against junk patterns
        for pattern in junk_patterns:
            if re.search(pattern, phrase_lower):
                return False

        # Must contain at least one substantive word (>3 chars)
        words = phrase.split()
        if not any(len(w) > 3 for w in words):
            return False

        return True

    def extract_keywords(self, text, k=15):
        """Extract noun phrases as keywords (filtered)"""
        if self.nlp is None:
            return []

        doc = self.nlp(text[:100000])  # Limit for speed

        # Extract noun chunks
        noun_chunks = []
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            if 1 <= len(phrase.split()) <= 3:  # 1-3 word phrases
                # Filter junk
                if self.is_valid_keyword(phrase):
                    noun_chunks.append(phrase)

        # Count frequency
        freq = Counter(noun_chunks)

        # Return top k (need to extract more to account for filtering)
        keywords = [phrase for phrase, count in freq.most_common(k*3)][:k]
        return keywords

    def extract_relations(self, text, keywords):
        """Extract relationships using dependency parsing"""
        if self.nlp is None:
            return []

        relations = []
        doc = self.nlp(text[:100000])  # Limit for speed

        keyword_set = set(kw.lower() for kw in keywords)

        # Extract relations based on dependency trees
        for token in doc:
            if token.text.lower() in keyword_set:
                # Check if head is also a keyword
                if token.head.text.lower() in keyword_set and token.text != token.head.text:
                    relations.append({
                        'source': token.text.lower(),
                        'relation': token.dep_.upper(),
                        'target': token.head.text.lower()
                    })

                # Check children
                for child in token.children:
                    if child.text.lower() in keyword_set and child.text != token.text:
                        relations.append({
                            'source': token.text.lower(),
                            'relation': child.dep_.upper(),
                            'target': child.text.lower()
                        })

        # Remove duplicates
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel['source'], rel['relation'], rel['target'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

    def process(self, text, k=15):
        """Main processing pipeline"""
        keywords = self.extract_keywords(text, k)
        relations = self.extract_relations(text, keywords)

        return {
            'nodes': keywords,
            'edges': relations
        }


class BaselineTFIDFOnly:
    """
    Baseline 3: TF-IDF Only keyword extraction
    No relation extraction (not applicable)
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_keywords(self, text, k=15):
        """Extract keywords using TF-IDF (with 1-3 word phrases)"""
        vectorizer = TfidfVectorizer(
            max_features=k*2,  # Get more candidates
            ngram_range=(1, 3),  # 1-3 word phrases (was 1-2)
            stop_words=list(self.stop_words),
            token_pattern=r'\b[a-z]{3,}\b'  # At least 3 chars
        )

        # Fit on single document
        try:
            X = vectorizer.fit_transform([text])
            candidates = vectorizer.get_feature_names_out()

            # Rank by TF-IDF scores
            scores = X.toarray()[0]
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

            # Return top k
            keywords = [phrase for phrase, score in ranked[:k]]
            return keywords
        except:
            return []

    def process(self, text, k=15):
        """Main processing pipeline"""
        keywords = self.extract_keywords(text, k)

        return {
            'nodes': keywords,
            'edges': []  # TF-IDF doesn't extract relations
        }


class BaselineYAKEOnly:
    """
    Baseline 4: YAKE Only keyword extraction
    No relation extraction (not applicable)
    """

    def __init__(self):
        pass

    def extract_keywords(self, text, k=15):
        """Extract keywords using YAKE (optimized for multi-word phrases)"""
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # Max 3-word phrases
            dedupLim=0.7,  # More aggressive deduplication (was 0.9)
            dedupFunc='seqm',  # Sequence matcher for deduplication
            windowsSize=2,  # Context window
            top=k*2  # Get more candidates
        )

        keywords = kw_extractor.extract_keywords(text)
        # Return top k (YAKE returns tuples of (keyword, score))
        return [kw.lower().strip() for kw, score in keywords[:k]]

    def process(self, text, k=15):
        """Main processing pipeline"""
        keywords = self.extract_keywords(text, k)

        return {
            'nodes': keywords,
            'edges': []  # YAKE doesn't extract relations
        }


# Utility function to run all baselines
def run_all_baselines(text, k=15):
    """
    Run all baseline methods on given text

    Returns:
        dict: Results from all baseline methods
    """
    results = {}

    # Co-occurrence
    print("Running Co-occurrence baseline...")
    cooccurrence = BaselineCooccurrence()
    results['cooccurrence'] = cooccurrence.process(text, k)

    # Rule-based
    print("Running Rule-based baseline...")
    rule_based = BaselineRuleBased()
    results['rule_based'] = rule_based.process(text, k)

    # TF-IDF Only
    print("Running TF-IDF Only baseline...")
    tfidf_only = BaselineTFIDFOnly()
    results['tfidf_only'] = tfidf_only.process(text, k)

    # YAKE Only
    print("Running YAKE Only baseline...")
    yake_only = BaselineYAKEOnly()
    results['yake_only'] = yake_only.process(text, k)

    return results


if __name__ == "__main__":
    # Test the baselines
    sample_text = """
    Sea ice extent in the Arctic Ocean has declined significantly since 1979.
    Temperature anomalies are strongly correlated with ice thickness reduction.
    Wind speed influences ocean currents, which in turn affect heat flux patterns.
    Albedo changes control the surface energy balance and melt rates.
    """

    results = run_all_baselines(sample_text, k=10)

    print("\n=== BASELINE RESULTS ===")
    for method, output in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Nodes: {output['nodes']}")
        print(f"  Edges: {len(output['edges'])} relationships")
