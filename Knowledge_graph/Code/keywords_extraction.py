import re, spacy, nltk, yake
import networkx as nx
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from fuzzywuzzy import fuzz
import torch
import ollama
from itertools import combinations
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")




## Extracting text from pdf using pdfplumber.
import pdfplumber
def text_extraction(path):
    input_text = ""
    with pdfplumber.open(path) as f:  ## pdfplumber is being used because of its nature to capture all the text in the pdf.
        for page in f.pages:
            input_text += page.extract_text() + "/n"

    return input_text

#print('Extracted_text:',text_extraction("arctic1.pdf"))

#input_text=text_extraction("arctic1.pdf")
#input_text



def cleaning_extracted_text(input_text):

  input_text= input_text.lower() # Lets convert the text into lowercase and go ahead for preprocessing
  # Lets remove extra sapces
  input_text = re.sub(r'\s+', ' ', input_text)

  # Remove urls
  input_text = re.sub(r"https?://\S+|www\.\S+", "", input_text)
  input_text = re.sub(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", "", input_text)
  input_text = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", input_text)

  # Lets remove the email headers
  input_text = re.sub(r"[\w.-]+@[\w.-]+\.\w+", "", input_text)
  # Lets remove the author names
  input_text = re.sub(r"(?i)(?:\n\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\([\w., ]+\)\s*\n?", "", input_text)

  # Let's work on clearing the references and in-text citations
  input_text = re.sub(r"(?i)References.*", "", input_text, flags=re.DOTALL)  # this removes the entire refeence section.

  # removing in-text citations
  input_text = re.sub(r"\[\d+\]", "", input_text)  # Removing citations like [2]

  input_text = re.sub(r"\(.*?et al\., \d{4}\)", "", input_text) # Removing intext citations.

  # Removing numeric data
  input_text = re.sub(r"\b\d+\b", "", input_text)
  # Lets remove punctuations
  input_text = re.sub(r'[^\w\s]', '', input_text)

  # Lets remove stopwords

  stopwords_list= stopwords.words('english')
  words_list=word_tokenize(input_text)
  words=[]
  cleaned_text=''
  for word in words_list:
    if word not in stopwords_list:
      words.append(word)
  cleaned_text+=' '.join(words)

  # Lets use lemmatozation
  final_input_text=""
  l = WordNetLemmatizer()
  for word in cleaned_text.split():
    final_input_text += l.lemmatize(word) + " "

  return final_input_text




def scale(s, method="default"):
    if not s:
        return []

    if method == "yake":
        s = [1 / (score + 1e-6) for score in s]

    log_s = np.log1p(s)
    exp_s = np.exp(log_s - np.max(log_s))
    softmax = exp_s / np.sum(exp_s)

    return softmax.tolist()
## Need to work on multiple keywords in multiple pdfs

stopwords_custom = set(stopwords.words('english')) | {"et", "al","also", "study","paper", "research", "abstract", "methodology","table",
                                                      "abstract", "introduction", "conclusion", "discussion", "results",
    "acknowledgments", "references", "et", "al", "doi", "journal",

    # Common measurement and mathematical terms
    "fig", "figure", "table", "equation", "calculated", "estimated", "observed",
    "km", "m", "cm", "hz", "measurement", "data", "analysis", "model", "method",
    "parameter", "frequency", "value", "observation", "estimate",

    # Common descriptive terms
    "high", "low", "increase", "decrease", "significant", "approximately",
    "large", "small", "compared", "higher", "lower", "greater", "less",

    # Common scientific/academic verbs
    "study", "research", "investigate", "analyze", "present", "provide",
    "describe", "report", "summarize", "conclude", "discuss", "demonstrate",

    # Paper-specific but non-informative terms
    "ship", "cruise", "instrument", "time", "period", "location",
    "university", "department", "author", "error", "deviation",

    # Common domain terms that would oversaturate results
    "wave", "sea", "ice", "ocean", "arctic", "model", "attenuation",
    "section", "fig", "table", "measured", "shown", "found", "used"}
## be careful with stopwords-custom, if possible make it to list before passing into models like keybert
''' check on input_text'''



def extract_keywords(input_text, k):
    input_text = input_text.lower()
    lines = input_text.splitlines()
    keyword_lines = []
    collecting = False

    for i, line in enumerate(lines):
        if "keywords" in line:  # no longer requiring ':'
            collecting = True
            keyword_lines.append(line.strip())
            # Check up to 2 more lines for continued keywords
            for j in range(i + 1, min(i + 3, len(lines))):
                next_line = lines[j].strip()
                # Stop if line looks like a heading
                if re.match(r"^\s*\d+[\.\)]?\s+[a-z]", next_line):  # e.g., 1. introduction
                    break
                if next_line == "":
                    continue
                keyword_lines.append(next_line)
            break

    list_keywords = []
    if keyword_lines:
        print("✅ Detected Multi-line or Single-line Keywords Section:")
        full_kw_line = " ".join(keyword_lines)
        full_kw_line = re.sub(r"(?i)keywords\s*[:\-]?\s*", "", full_kw_line)  # remove 'keywords:'
        full_kw_line = full_kw_line.replace("·", ",").replace("•", ",").replace(";", ",")
        full_kw_line = re.sub(r"[^\w,\-\s]", "", full_kw_line)
        list_keywords = [kw.strip() for kw in full_kw_line.split(",") if kw.strip()]
        print(f"Extracted from Keywords section: {list_keywords}")
        return list_keywords[:k] if len(list_keywords) > k else list_keywords

    # TF-IDF
    input_text=cleaning_extracted_text(input_text)
    list_documents = [input_text]
    tf_vector = TfidfVectorizer(
        stop_words=list(stopwords_custom), # Removing custom stopwords
        ngram_range=(1,2),  # Including unigrams and bigrams keywords
        sublinear_tf=True)  # this removes the bias towards most frequent words

    # Creating a matrix of words with their scores
    tf_matrix = tf_vector.fit_transform(list_documents)

    words = tf_vector.get_feature_names_out()
    values = tf_matrix.toarray().sum(axis=0)

    # lets sort the keywords in descending order of their values
    tf_scores = sorted(zip(words, values), key=lambda x: x[1], reverse=True)

    tfidf_keywords= tf_scores[:k]

    # Normalising the scores
    sc = [s for _, s in tfidf_keywords]
    norm_scores = scale(sc,'tf-idf')

    tfidf_keywords = [(keyword.lower().strip(), l) for (keyword, _), l in zip(tfidf_keywords, norm_scores)]



    # YAKE
    # Lets combine all pdf text into single text
    if len(list_documents)>1:
      input_text=''
      for i in list_documents:
        input_text+=i.lower()
    else:

  # As Yake only extracts either unigrams or bigrams but not combinely, lets combine them
      u = yake.KeywordExtractor(lan="en", n=1, top=k//2)
      uni_keywords = u.extract_keywords(input_text)

    # Bigrams extraction
      bi = yake.KeywordExtractor(lan="en", n=2, top=k//2)
      bi_keywords = bi.extract_keywords(input_text)

    # Let us extract scores for normalization
      uni_scores = [s for i, s in uni_keywords]
      bi_scores = [s for j, s in bi_keywords]


    # Assign the normalized scores back to keywords
      uni_keywords = [(key.lower().strip(), n) for (key, _), n in zip(uni_keywords, scale(uni_scores,'yake'))]
      bi_keywords = [(key.lower().strip(), n) for (key, _), n in zip(bi_keywords, scale(bi_scores,'yake'))]

    # Combining unigram and bigrams
      yake_keywords = uni_keywords + bi_keywords

    # Sort by scaled score (lower is better)
      yake_keywords = sorted(yake_keywords, key=lambda x: x[1])


    # KEYBERT
    model = SentenceTransformer("all-MiniLM-L6-v2")
    keybert_model = KeyBERT(model=model)

    # Let's extract unigrams and bigrams separately and then merge them
    uni_keywords = keybert_model.extract_keywords(
        input_text,
        keyphrase_ngram_range=(1, 1),
        stop_words=list(stopwords_custom),
        top_n=k//2,
        use_mmr=True,
        diversity=0.3
    )


    bi_keywords = keybert_model.extract_keywords(
        input_text,
        keyphrase_ngram_range=(2, 2),
        stop_words=list(stopwords_custom),
        top_n=k//2,
        use_mmr=True,
        diversity=0.3
    )


    # Normalization
    uni_score = [s for _, s in uni_keywords]
    bi_score = [s for _, s in bi_keywords]

    norm_uni_scores = scale(uni_score, 'keybert')  # Scaling unigram scores
    norm_bi_scores = scale(bi_score, 'keybert')  # Scaling bigram scores

    # Assign normalized scores back to keywords
    uni_keywords = [(key.lower().strip(), s) for (key, _), s in zip(uni_keywords, norm_uni_scores)]
    bi_keywords = [(key.lower().strip(), s) for (key, _), s in zip(bi_keywords, norm_bi_scores)]

# Merge the results after separate scaling
    keybert_keywords = sorted(uni_keywords + bi_keywords, key=lambda x: x[1], reverse=True)

# Lets combine all the methods keywords and return the important ones

    d = defaultdict(list)

    # Let's append all the keywords and their normalized scores.
    for key, score in tfidf_keywords:
        key = str(key).lower().strip()
        d[key].append(score*0.25)

    for key, score in yake_keywords:
        key = str(key).lower().strip()
        d[key].append(score*0.5)

    for key, score in keybert_keywords:
        key = str(key).lower().strip()
        d[key].append(score*0.25)

    scores = {key: sum(scores) for key, scores in d.items()}

# Sort the scores in descending order
    combined_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Check if we have fewer than k unique keywords
    if len(combined_keywords) < k:
        print(f"⚠️ Warning: Only {len(combined_keywords)} unique keywords could be extracted, "
              f"though you requested {k}. The text may be too short or keyword overlaps occurred.")

# Let's take top k keywords
    for i, j in combined_keywords[:k]:
        list_keywords.append(i)
        print((i, j))

    print('The extracted Keywords are :', list_keywords)
    return list_keywords




def extract_all_keyword_pairs(keywords):
    return list(combinations(sorted(set(keywords)), 2))

def text_chunks(input_text, chunk_size=2000, overlap=300):
    chunks = []
    i = 0
    while i < len(input_text):
        chunks.append(input_text[i:i + chunk_size])
        i += chunk_size - overlap
    return chunks
#chunks = text_chunks(input_text)


import re
import ollama

def extract_relations_llama_all(text, keyword_pairs):
    prompt = f"""
You are an expert assistant trained to extract semantic relationships from Arctic and climate science research papers.

Your task is to analyze the following scientific text and infer meaningful relationships between each keyword pair listed below.

TEXT:
{text}

KEYWORD PAIRS:
{', '.join([f'("{a}", "{b}")' for a, b in keyword_pairs])}

INSTRUCTIONS:

1. Output format:
   (KEYWORD_1, RELATION, KEYWORD_2)

2. Only use the provided keywords in keywords_pairs exactly as they appear.
   - Do NOT invent new keywords or alter the phrases.

3. RELATION format rules:
   - Must be in **UPPERCASE**
   - No spaces, lowercase letters, hyphens, or symbols
   - Use underscores instead of spaces if needed

4. Use precise domain-specific relations when possible.
   - Examples: MEASURED_BY, MODULATES, CAUSED_BY, INTERACTS_WITH, TRACKED_WITH
   - Avoid overly generic ones like AFFECTS unless truly necessary.

5. If no clear relation is found, use: RELATED_TO

6. Do NOT include explanations, summaries, or extra commentary.
7. If any output line contains keywords that are NOT from the provided list, discard that line.

8. Do NOT include notes, summaries, explanations, or invented phrases. Output ONLY relation triples.


EXAMPLE OUTPUT:
(wave height, MEASURED_BY, altimeter)
(sea ice, INTERACTS_WITH, ocean current)

Now return one relation per line in this exact format:
(KEYWORD_1, RELATION, KEYWORD_2)
"""
    # Call Ollama
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    content = response['message']['content']

    # Extract relations using regex
    extracted = []
    for line in content.strip().split("\n"):
        match = re.match(r"\(?([^,]+),\s*([A-Z0-9_]+),\s*([^)]+)\)?", line.strip())
        if match:
            head, relation, tail = match.groups()
            extracted.append((head.strip(), relation.strip(), tail.strip()))
    return extracted

def confidence_scores(relations, embed_model):
    final = []
    for head, rel, tail in relations:
        emb = embed_model.encode([head + " " + rel, tail])
        score = cosine_similarity([emb[0]], [emb[1]])[0][0]
        final.append((head, rel, tail, round(float(score), 3)))
    return final
def cosine_score(h, r, t,model):
    emb = model.encode([h + " " + r, t])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def hybrid_score(h, r, t,model):
    # Cosine similarity
    emb = model.encode([h + " " + r, t])
    cos_score = float(cosine_similarity([emb[0]], [emb[1]])[0][0])

        # Fuzzy score between head+rel and tail
    fuzz_score = fuzz.token_sort_ratio(h + " " + r, t) / 100.0

        # Bonus if common syntactic relation terms are used
    direction_bonus = 0.1 if r in ["causes", "measures", "defines", "predicts", "relates to", "uses"] else 0

        # Weighted sum
    return round(0.6 * cos_score + 0.3 * fuzz_score + direction_bonus, 3)



def process(file_path, k):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    input_text = text_extraction(file_path)
    print('Input text checking:', input_text[:3000])

    # Step 0: Extract keywords
    keywords = extract_keywords(input_text, k)
    print(f"\nExtracted Keywords are: {keywords}")

    # Normalize for consistency
    filtered_candidates = [kw.lower().strip() for kw in keywords]
    print(f"\nFiltered Keywords: {filtered_candidates}")

    # Step 1: Chunk the text
    chunks = text_chunks(input_text)

    # Step 2: Use LLaMA to generate relations from each chunk
    total_relations = []
    valid_keywords = set(filtered_candidates)

    for c in chunks:
        keyword_pairs = extract_all_keyword_pairs(filtered_candidates)
        extracted = extract_relations_llama_all(c, keyword_pairs)

        # Filter hallucinated or off-topic relations
        filtered_relations = [
            (h, r, t)
            for h, r, t in extracted
            if h.lower().strip() in valid_keywords and t.lower().strip() in valid_keywords
        ]
        total_relations.extend(filtered_relations)

    # Step 3: Score relations
    scored = confidence_scores(total_relations, model)

    print("\n\nRelations with Confidence Scores:\n")
    for h, r, t, s in sorted(scored, key=lambda x: -x[3]):
        print(f"({h}, {r}, {t}) : {s}")

    # Compare Cosine and Hybrid Scores
    print("\nComparison of Cosine Score and Hybrid Score:\n")
    for h, r, t in total_relations:
        cos = round(cosine_score(h, r, t, model), 3)
        hybrid = hybrid_score(h, r, t, model)
        print(f"({h}, {r}, {t})")
        print(f"   - Cosine Similarity Score: {cos}")
        print(f"   - Hybrid Score           : {hybrid}\n")

    # Step 4: Prepare top relations for Neo4j
    top_scored_relations = sorted(scored, key=lambda x: -x[3])[:100]

    neo4j_nodes = set()
    neo4j_edges = []

    for head, rel, tail, score in top_scored_relations:
        neo4j_nodes.add(head)
        neo4j_nodes.add(tail)
        neo4j_edges.append({
            "source": head,
            "relation": rel,
            "target": tail,
            "score": score
        })

    return list(neo4j_nodes), neo4j_edges
