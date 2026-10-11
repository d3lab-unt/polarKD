import re
import ollama
from pyvis.network import Network

# Words that indicate a node is a sentence fragment from the paper, not a variable
_SENTENCE_STARTERS = {
    'abstract', 'this', 'the', 'we', 'our', 'in', 'here', 'study',
    'paper', 'article', 'introduction', 'results', 'conclusion',
    'figure', 'table', 'section', 'however', 'therefore', 'although',
    'based', 'using', 'used', 'proposed', 'present', 'show', 'shown',
    'demonstrate', 'investigate', 'analyse', 'analyze', 'evaluate',
}

# Computational / ML / method terms that are never research variables
_TECH_BLOCKLIST = {
    # ML model families
    'machine learning', 'deep learning', 'neural network', 'neural networks',
    'random forest', 'decision tree', 'decision trees', 'gradient boosting',
    'xgboost', 'lightgbm', 'catboost', 'support vector machine', 'svm',
    'naive bayes', 'k-nearest neighbor', 'knn',
    # DL architectures
    'lstm', 'gru', 'rnn', 'cnn', 'transformer', 'bert', 'gpt',
    'convolutional neural network', 'recurrent neural network',
    'multilayer perceptron', 'mlp', 'autoencoder', 'gan', 'vae',
    'attention mechanism', 'self-attention', 'encoder', 'decoder',
    # Training / optimisation
    'backpropagation', 'gradient descent', 'stochastic gradient',
    'learning rate', 'batch size', 'epoch', 'epochs', 'hyperparameter',
    'dropout', 'regularization', 'batch normalization', 'layer normalization',
    'overfitting', 'underfitting', 'early stopping', 'weight decay',
    # Data / feature engineering
    'feature extraction', 'feature selection', 'feature engineering',
    'dimensionality reduction', 'principal component analysis', 'pca',
    'data augmentation', 'data preprocessing', 'normalization',
    'train test split', 'cross validation', 'k-fold',
    'transfer learning', 'fine-tuning', 'pre-training',
    # Evaluation metrics
    'accuracy', 'precision', 'recall', 'f1 score', 'f1-score',
    'rmse', 'mae', 'mse', 'r-squared', 'auc', 'roc curve',
    'mean absolute error', 'root mean square error', 'mean squared error',
    'confusion matrix', 'classification report',
    # General computational terms
    'algorithm', 'model architecture', 'hyperparameter tuning',
    'embedding', 'tokenization', 'vectorization',
    'inference', 'prediction model', 'forecasting model',
}


def _is_valid_node(text: str) -> bool:
    """Return True only if a node looks like a research variable, not a method or sentence."""
    t = text.strip()
    if not t:
        return False

    # NEW: remove leading articles instead of rejecting useful variable names
    # Example: "the sea ice extent" -> "sea ice extent"
    t = re.sub(r'^(the|a|an)\s+', '', t, flags=re.IGNORECASE).strip()

    # Reject anything longer than 60 characters
    if len(t) > 60:
        return False

    # Reject anything with more than 5 words
    words = t.split()
    if len(words) > 5:
        return False

    # Reject if the first word is a known sentence-starter
    if words[0].lower() in _SENTENCE_STARTERS:
        return False

    # Reject if node text matches or contains any technical/method term
    t_lower = t.lower()
    if any(term in t_lower for term in _TECH_BLOCKLIST):
        return False

    return True

def _clean_node(text: str) -> str:
    return text.strip().lower()


def extract_causal_relations(kg_edges: list, model: str = "mistral") -> list:
    """
    Pass 2: receives KG edges (list of dicts with source/relation/target/score)
    and asks the LLM to identify which ones are causal.

    Three layers of protection against noise:
      1. Pre-filter: drop any edge whose source or target fails _is_valid_node()
      2. Prompt constraint: send the LLM a whitelist of valid node names
      3. Post-validate: discard any parsed CAUSE/EFFECT not in the whitelist

    Returns list of dicts: {cause, effect, label, confidence}
    """
    if not kg_edges:
        return []

    # ── Layer 1: pre-filter noisy edges ──────────────────────────────────
    clean_edges = [
        e for e in kg_edges
        if _is_valid_node(e.get('source', '')) and _is_valid_node(e.get('target', ''))
    ]

    if not clean_edges:
        print("⚠️  All KG edges were filtered out as noisy. Check keyword extraction quality.")
        return []

    # Deduplicate (same triple can appear from multiple chunks)
    seen = set()
    deduped = []
    for e in clean_edges:
        key = (e['source'].lower().strip(), e['relation'], e['target'].lower().strip())
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # ── Build valid node whitelist ────────────────────────────────────────
    valid_nodes = sorted({
        _clean_node(e['source']) for e in deduped
    } | {
        _clean_node(e['target']) for e in deduped
    })
    valid_nodes_lower = set(valid_nodes)

    nodes_str = "\n".join(f"  - {n}" for n in valid_nodes)
    edges_str = "\n".join(
        f"({e['source']}, {e['relation']}, {e['target']})"
        for e in deduped
    )

    # ── Layer 2: constrained prompt ───────────────────────────────────────
    prompt = f"""You are an expert in causal reasoning for climate, Arctic, and Earth sciences.

DOMAIN CONTEXT: This paper is from climate / environmental / Earth science research.
You are identifying causal relationships between PHYSICAL and ENVIRONMENTAL variables only.

VALID VARIABLE NAMES (copy VERBATIM — do not paraphrase, abbreviate, or invent):
{nodes_str}

KNOWLEDGE GRAPH EDGES to analyse:
{edges_str}

TASK: Identify edges where one physical/environmental variable directly causes, drives, triggers, or produces change in another.

ACCEPT pairs where CAUSE and EFFECT are:
- Physical quantities (temperature, pressure, salinity, humidity)
- Environmental phenomena (sea ice, permafrost, albedo, precipitation)
- Biogeochemical processes (carbon flux, evaporation, photosynthesis)
- Climate indices or forcings (radiative forcing, ENSO, heat flux)
- Measured dataset variables (SST, SIC, SLP, OHC, AOD)

REJECT any pair where CAUSE or EFFECT is:
- A computational method (machine learning, neural network, LSTM, regression)
- A model architecture or algorithm name
- A statistical technique or evaluation metric (RMSE, accuracy, R-squared)
- A software tool, framework, or dataset name (not a variable measured by it)
- A general research activity (training, prediction, classification)

For each accepted causal pair output EXACTLY (one blank line between blocks):

CAUSE: <exact string from the list>
EFFECT: <exact string from the list>
LABEL: <mechanism in UPPERCASE_WITH_UNDERSCORES>
CONFIDENCE: <0.0 to 1.0>

Rules:
- CAUSE and EFFECT must be VERBATIM from the list above
- You may reverse direction if the relation implies it (e.g. "B CAUSED_BY A" → CAUSE=A, EFFECT=B)
- LABEL examples: DRIVES, CAUSES, LEADS_TO, TRIGGERS, AMPLIFIES, INHIBITS, ACCELERATES, REDUCES, WARMS, MELTS, INCREASES, DECREASES
- CONFIDENCE: 1.0 = direct physical mechanism | 0.6 = established link | 0.3 = indirect / uncertain
- Skip: CORRELATES_WITH, IS_A, MEASURED_BY, RELATED_TO, IS_USED_FOR, IS_PART_OF
- Output ONLY the blocks — no explanations, headers, or extra text

Now identify all causal relationships:
"""

    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        content = response['message']['content']
        # ── Layer 3: post-validate against whitelist + confidence threshold ─
        return _parse_causal_output(content, valid_nodes_lower, min_confidence=0.4)
    except Exception as e:
        print(f"Causal extraction error: {e}")
        return []


def _parse_causal_output(
    text: str,
    valid_nodes_lower: set | None = None,
    min_confidence: float = 0.4,
) -> list:
    """
    Parse the LLM causal output into structured dicts.
    Discards pairs whose nodes are not in valid_nodes_lower (hallucinations),
    and pairs below min_confidence (weak/uncertain links).
    """
    results = []
    blocks = re.split(r'\n\s*\n', text.strip())

    for block in blocks:
        cause = effect = label = None
        confidence = 0.5

        for line in block.strip().split('\n'):
            line = line.strip()
            key = line.upper()
            if key.startswith('CAUSE:'):
                cause = line.split(':', 1)[1].strip()
            elif key.startswith('EFFECT:'):
                effect = line.split(':', 1)[1].strip()
            elif key.startswith('LABEL:'):
                raw = line.split(':', 1)[1].strip()
                label = re.sub(r'[^A-Z0-9_]', '_', raw.upper()).strip('_')
            elif key.startswith('CONFIDENCE:'):
                match = re.search(r'[\d.]+', line.split(':', 1)[1])
                if match:
                    try:
                        confidence = max(0.0, min(1.0, float(match.group())))
                    except ValueError:
                        confidence = 0.5

        if not (cause and effect and label):
            continue

        # Post-validate against whitelist
        if valid_nodes_lower is not None:
            if cause.lower() not in valid_nodes_lower or effect.lower() not in valid_nodes_lower:
                print(f"⚠️  Rejected hallucinated pair: '{cause}' → '{effect}'")
                continue

        # Skip self-loops
        if cause.lower() == effect.lower():
            continue

        # Drop low-confidence pairs (likely method/correlation noise)
        if confidence < min_confidence:
            print(f"⚠️  Rejected low-confidence pair ({confidence}): '{cause}' → '{effect}'")
            continue

        results.append({
            'cause': cause,
            'effect': effect,
            'label': label,
            'confidence': round(confidence, 2)
        })

    return results


def _confidence_to_color(confidence: float) -> str:
    if confidence >= 0.8:
        return '#C0392B'
    elif confidence >= 0.6:
        return '#E74C3C'
    elif confidence >= 0.4:
        return '#E67E22'
    else:
        return '#F39C12'


def generate_causal_graph(causal_relations: list) -> tuple:
    """
    Build a directed PyVis causal graph from causal relation dicts.
    Returns (Network, html_string).
    """
    net = Network(height="500px", width="100%", directed=True,
                  bgcolor="#0D1117", font_color="#FFFFFF")

    net.set_options("""{
        "physics": {
            "enabled": true,
            "repulsion": {
                "nodeDistance": 250,
                "springLength": 300,
                "springConstant": 0.04
            },
            "solver": "repulsion"
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 1.2}},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
            "font": {"size": 11, "color": "#CCCCCC", "strokeWidth": 0}
        },
        "nodes": {
            "font": {"size": 13, "color": "#FFFFFF"},
            "borderWidth": 2
        }
    }""")

    if not causal_relations:
        return net, ""

    cause_count = {}
    effect_count = {}
    for rel in causal_relations:
        cause_count[rel['cause']] = cause_count.get(rel['cause'], 0) + 1
        effect_count[rel['effect']] = effect_count.get(rel['effect'], 0) + 1

    all_nodes = set(r['cause'] for r in causal_relations) | set(r['effect'] for r in causal_relations)

    for node in all_nodes:
        is_cause = node in cause_count
        is_effect = node in effect_count

        if is_cause and is_effect:
            color = {"background": "#8E44AD", "border": "#6C3483"}
        elif is_cause:
            color = {"background": "#C0392B", "border": "#922B21"}
        else:
            color = {"background": "#E67E22", "border": "#CA6F1E"}

        size = 20 + (cause_count.get(node, 0) * 5) + (effect_count.get(node, 0) * 3)
        size = min(size, 50)

        causes_list = ", ".join(r['effect'] for r in causal_relations if r['cause'] == node)
        caused_by_list = ", ".join(r['cause'] for r in causal_relations if r['effect'] == node)
        tooltip = f"<b>{node}</b>"
        if causes_list:
            tooltip += f"<br>Causes: {causes_list}"
        if caused_by_list:
            tooltip += f"<br>Caused by: {caused_by_list}"

        net.add_node(node, label=node, color=color, size=size, title=tooltip)

    for rel in causal_relations:
        edge_color = _confidence_to_color(rel['confidence'])
        width = round(1 + rel['confidence'] * 3, 1)
        net.add_edge(
            rel['cause'],
            rel['effect'],
            label=rel['label'],
            color=edge_color,
            width=width,
            title=f"{rel['label']} (confidence: {rel['confidence']})"
        )

    net.save_graph("causal_graph.html")
    with open("causal_graph.html", "r") as f:
        html_str = f.read()
    return net, html_str
