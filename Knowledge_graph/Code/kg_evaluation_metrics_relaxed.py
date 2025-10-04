"""
Knowledge Graph Evaluation Metrics - RELAXED VERSION
Implements fuzzy matching for node names to handle terminology variations
"""

import networkx as nx
from typing import Dict, List, Set, Tuple
from fuzzywuzzy import fuzz


class KGEvaluatorRelaxed:
    """
    Relaxed Knowledge Graph Evaluator with fuzzy node matching
    Handles variations in terminology (e.g., "sea ice concentration" vs "ice concentration")
    """

    def __init__(self, node_similarity_threshold=80):
        """
        Args:
            node_similarity_threshold: Fuzzy match threshold for nodes (0-100)
                80 = "sea ice concentration" matches "ice concentration" (80% similar)
                90 = Stricter matching
                70 = More lenient matching
        """
        self.node_threshold = node_similarity_threshold

    def fuzzy_match_nodes(self, gold_nodes: List[str], pred_nodes: List[str]) -> Dict:
        """
        Match nodes with fuzzy string matching
        Returns mapping of gold_node -> best_matching_pred_node
        """
        matches = {}
        matched_pred = set()

        for gold_node in gold_nodes:
            best_match = None
            best_score = 0

            for pred_node in pred_nodes:
                if pred_node in matched_pred:
                    continue

                # Fuzzy match score
                score = fuzz.token_sort_ratio(gold_node.lower(), pred_node.lower())

                if score > best_score and score >= self.node_threshold:
                    best_score = score
                    best_match = pred_node

            if best_match:
                matches[gold_node.lower()] = best_match.lower()
                matched_pred.add(best_match.lower())

        return matches

    def compute_node_f1(self, gold_nodes: List[str], pred_nodes: List[str]) -> Dict:
        """
        Compute Node-level F1 score with fuzzy matching
        """
        # Fuzzy match nodes
        matches = self.fuzzy_match_nodes(gold_nodes, pred_nodes)

        tp = len(matches)
        fp = len([p for p in pred_nodes if p.lower() not in matches.values()])
        fn = len([g for g in gold_nodes if g.lower() not in matches.keys()])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'matches': matches,
            'threshold_used': self.node_threshold
        }

    def map_edge_nodes(self, edge: Dict, node_mapping: Dict) -> Dict:
        """
        Map edge nodes using fuzzy node mapping
        """
        source = edge['source'].lower()
        target = edge['target'].lower()

        # Map to matched nodes
        mapped_source = node_mapping.get(source, source)
        mapped_target = node_mapping.get(target, target)

        return {
            'source': mapped_source,
            'relation': edge['relation'].upper(),
            'target': mapped_target
        }

    def compute_edge_f1_strict(self, gold_edges: List[Dict], pred_edges: List[Dict],
                                gold_nodes: List[str], pred_nodes: List[str]) -> Dict:
        """
        Compute Edge-level F1 score with node fuzzy matching
        """
        # Get node mapping
        node_mapping = self.fuzzy_match_nodes(gold_nodes, pred_nodes)

        # Create reverse mapping for pred -> gold
        reverse_mapping = {v: k for k, v in node_mapping.items()}

        # Map edges
        def normalize_edge(edge, mapping):
            source = edge['source'].lower()
            target = edge['target'].lower()

            # Map nodes
            mapped_source = mapping.get(source, source)
            mapped_target = mapping.get(target, target)

            return (mapped_source, edge['relation'].upper(), mapped_target)

        # Normalize gold edges (keep as is)
        gold_set = set(normalize_edge(e, {n.lower(): n.lower() for n in gold_nodes}) for e in gold_edges)

        # Normalize pred edges (map to gold node names)
        pred_set = set(normalize_edge(e, reverse_mapping) for e in pred_edges)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def compute_edge_f1_relaxed(self, gold_edges: List[Dict], pred_edges: List[Dict],
                                 gold_nodes: List[str], pred_nodes: List[str]) -> Dict:
        """
        Compute Edge-level F1 score with relaxed matching (ignore relation type)
        """
        # Get node mapping
        node_mapping = self.fuzzy_match_nodes(gold_nodes, pred_nodes)
        reverse_mapping = {v: k for k, v in node_mapping.items()}

        def get_node_pair(edge, mapping):
            source = edge['source'].lower()
            target = edge['target'].lower()

            mapped_source = mapping.get(source, source)
            mapped_target = mapping.get(target, target)

            return tuple(sorted([mapped_source, mapped_target]))

        gold_pairs = set(get_node_pair(e, {n.lower(): n.lower() for n in gold_nodes}) for e in gold_edges)
        pred_pairs = set(get_node_pair(e, reverse_mapping) for e in pred_edges)

        tp = len(gold_pairs & pred_pairs)
        fp = len(pred_pairs - gold_pairs)
        fn = len(gold_pairs - pred_pairs)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def compute_macro_f1(self, node_f1: float, edge_f1_strict: float) -> Dict:
        """Compute Macro-averaged F1 score"""
        macro_f1 = (node_f1 + edge_f1_strict) / 2
        return {'macro_f1': macro_f1}

    def evaluate(self, gold_graph: Dict, pred_graph: Dict) -> Dict:
        """
        Complete evaluation with fuzzy node matching
        """
        results = {}

        # 1. Node F1 (with fuzzy matching)
        node_metrics = self.compute_node_f1(gold_graph['nodes'], pred_graph['nodes'])
        results['node'] = node_metrics

        # 2. Edge F1 (Strict - with fuzzy node matching)
        edge_strict_metrics = self.compute_edge_f1_strict(
            gold_graph['edges'], pred_graph['edges'],
            gold_graph['nodes'], pred_graph['nodes']
        )
        results['edge_strict'] = edge_strict_metrics

        # 3. Edge F1 (Relaxed - with fuzzy node matching)
        edge_relaxed_metrics = self.compute_edge_f1_relaxed(
            gold_graph['edges'], pred_graph['edges'],
            gold_graph['nodes'], pred_graph['nodes']
        )
        results['edge_relaxed'] = edge_relaxed_metrics

        # 4. Macro F1
        macro_metrics = self.compute_macro_f1(node_metrics['f1'], edge_strict_metrics['f1'])
        results['macro'] = macro_metrics

        return results

    def print_report(self, results: Dict, method_name: str = "Method"):
        """Print evaluation report"""
        print("=" * 70)
        print(f"KNOWLEDGE GRAPH EVALUATION: {method_name}")
        print("=" * 70)

        # Macro F1
        print("\n📊 UNIFIED METRIC (Macro-Averaged F1)")
        print(f"  **Macro F1 Score: {results['macro']['macro_f1']:.4f}** ({results['macro']['macro_f1']:.2%})")

        # Node metrics
        print("\n📍 Component: Node Extraction (Fuzzy Matching)")
        print(f"  Similarity Threshold: {results['node']['threshold_used']}%")
        print(f"  Precision: {results['node']['precision']:.4f} ({results['node']['precision']:.2%})")
        print(f"  Recall:    {results['node']['recall']:.4f} ({results['node']['recall']:.2%})")
        print(f"  F1 Score:  {results['node']['f1']:.4f} ({results['node']['f1']:.2%})")
        print(f"  (TP={results['node']['tp']}, FP={results['node']['fp']}, FN={results['node']['fn']})")

        # Edge metrics (Strict)
        print("\n🔗 Component: Edge Extraction (Strict - Relation Type Must Match)")
        print(f"  Precision: {results['edge_strict']['precision']:.4f} ({results['edge_strict']['precision']:.2%})")
        print(f"  Recall:    {results['edge_strict']['recall']:.4f} ({results['edge_strict']['recall']:.2%})")
        print(f"  F1 Score:  {results['edge_strict']['f1']:.4f} ({results['edge_strict']['f1']:.2%})")
        print(f"  (TP={results['edge_strict']['tp']}, FP={results['edge_strict']['fp']}, FN={results['edge_strict']['fn']})")

        # Edge metrics (Relaxed)
        print("\n🔗 Component: Edge Extraction (Relaxed - Any Relation)")
        print(f"  Precision: {results['edge_relaxed']['precision']:.4f} ({results['edge_relaxed']['precision']:.2%})")
        print(f"  Recall:    {results['edge_relaxed']['recall']:.4f} ({results['edge_relaxed']['recall']:.2%})")
        print(f"  F1 Score:  {results['edge_relaxed']['f1']:.4f} ({results['edge_relaxed']['f1']:.2%})")
        print(f"  (TP={results['edge_relaxed']['tp']}, FP={results['edge_relaxed']['fp']}, FN={results['edge_relaxed']['fn']})")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test with fuzzy matching
    gold_graph = {
        'nodes': ['sea ice concentration', 'wave attenuation', 'ice thickness'],
        'edges': [
            {'source': 'sea ice concentration', 'relation': 'CONTROLS', 'target': 'wave attenuation'}
        ]
    }

    pred_graph = {
        'nodes': ['ice concentration', 'wave attenuation', 'thickness'],  # Slightly different names
        'edges': [
            {'source': 'ice concentration', 'relation': 'CONTROLS', 'target': 'wave attenuation'}
        ]
    }

    evaluator = KGEvaluatorRelaxed(node_similarity_threshold=80)
    results = evaluator.evaluate(gold_graph, pred_graph)
    evaluator.print_report(results, "Test Method")
