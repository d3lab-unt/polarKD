"""
Knowledge Graph Evaluation Metrics
Implements: Macro F1, Node F1, Edge F1 (Strict/Relaxed), GED
"""

import networkx as nx
from typing import Dict, List, Set, Tuple


class KGEvaluator:
    """
    Comprehensive Knowledge Graph Evaluator
    Computes Macro F1, Node F1, Edge F1 (strict/relaxed), and Graph Edit Distance
    """

    def __init__(self):
        pass

    def compute_node_f1(self, gold_nodes: List[str], pred_nodes: List[str]) -> Dict:
        """
        Compute Node-level F1 score (keyword extraction quality)

        Args:
            gold_nodes: Ground truth keywords
            pred_nodes: Predicted keywords

        Returns:
            dict: precision, recall, f1, tp, fp, fn
        """
        gold_set = set(n.lower().strip() for n in gold_nodes)
        pred_set = set(n.lower().strip() for n in pred_nodes)

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

    def compute_edge_f1_strict(self, gold_edges: List[Dict], pred_edges: List[Dict]) -> Dict:
        """
        Compute Edge-level F1 score with strict matching
        (source, relation, target) must all match exactly

        Args:
            gold_edges: Ground truth edges
            pred_edges: Predicted edges

        Returns:
            dict: precision, recall, f1
        """

        def normalize_edge(edge):
            return (
                edge['source'].lower().strip(),
                edge['relation'].upper().strip(),
                edge['target'].lower().strip()
            )

        gold_set = set(normalize_edge(e) for e in gold_edges)
        pred_set = set(normalize_edge(e) for e in pred_edges)

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

    def compute_edge_f1_relaxed(self, gold_edges: List[Dict], pred_edges: List[Dict]) -> Dict:
        """
        Compute Edge-level F1 score with relaxed matching
        Only (source, target) pair needs to match, relation type ignored

        Args:
            gold_edges: Ground truth edges
            pred_edges: Predicted edges

        Returns:
            dict: precision, recall, f1
        """

        def get_node_pair(edge):
            s = edge['source'].lower().strip()
            t = edge['target'].lower().strip()
            # Sort to make undirected
            return tuple(sorted([s, t]))

        gold_pairs = set(get_node_pair(e) for e in gold_edges)
        pred_pairs = set(get_node_pair(e) for e in pred_edges)

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
        """
        Compute Macro-averaged F1 score
        Simple average of Node F1 and Edge F1 (strict)

        Args:
            node_f1: Node F1 score
            edge_f1_strict: Edge F1 score (strict)

        Returns:
            dict: macro_f1
        """
        macro_f1 = (node_f1 + edge_f1_strict) / 2

        return {
            'macro_f1': macro_f1
        }

    def compute_graph_edit_distance(self, gold_edges: List[Dict], pred_edges: List[Dict]) -> Dict:
        """
        Compute Graph Edit Distance (GED)
        Number of operations needed to transform predicted graph to gold graph

        Args:
            gold_edges: Ground truth edges
            pred_edges: Predicted edges

        Returns:
            dict: ged, normalized_ged, similarity
        """
        # Build NetworkX graphs
        G_gold = nx.DiGraph()
        G_pred = nx.DiGraph()

        # Add edges (nodes added automatically)
        for edge in gold_edges:
            G_gold.add_edge(
                edge['source'].lower().strip(),
                edge['target'].lower().strip(),
                relation=edge['relation'].upper().strip()
            )

        for edge in pred_edges:
            G_pred.add_edge(
                edge['source'].lower().strip(),
                edge['target'].lower().strip(),
                relation=edge['relation'].upper().strip()
            )

        # Compute GED
        try:
            # For small graphs (<20 nodes), exact computation
            if len(G_gold.nodes()) <= 20 and len(G_pred.nodes()) <= 20:
                ged = nx.graph_edit_distance(G_gold, G_pred, timeout=30)
            else:
                # For larger graphs, use approximation
                ged = nx.graph_edit_distance(G_gold, G_pred, timeout=10)

            # Normalize by max graph size
            max_elements = max(
                len(G_gold.nodes()) + len(G_gold.edges()),
                len(G_pred.nodes()) + len(G_pred.edges())
            )

            normalized_ged = ged / max_elements if max_elements > 0 else 0
            similarity = 1 - normalized_ged

            return {
                'ged': ged,
                'normalized_ged': normalized_ged,
                'similarity': similarity
            }
        except:
            # If GED computation fails, return None
            return {
                'ged': None,
                'normalized_ged': None,
                'similarity': None
            }

    def evaluate(self, gold_graph: Dict, pred_graph: Dict) -> Dict:
        """
        Complete evaluation of knowledge graph

        Args:
            gold_graph: {'nodes': [...], 'edges': [...]}
            pred_graph: {'nodes': [...], 'edges': [...]}

        Returns:
            dict: All evaluation metrics
        """
        results = {}

        # 1. Node F1
        node_metrics = self.compute_node_f1(gold_graph['nodes'], pred_graph['nodes'])
        results['node'] = node_metrics

        # 2. Edge F1 (Strict)
        edge_strict_metrics = self.compute_edge_f1_strict(gold_graph['edges'], pred_graph['edges'])
        results['edge_strict'] = edge_strict_metrics

        # 3. Edge F1 (Relaxed)
        edge_relaxed_metrics = self.compute_edge_f1_relaxed(gold_graph['edges'], pred_graph['edges'])
        results['edge_relaxed'] = edge_relaxed_metrics

        # 4. Macro F1
        macro_metrics = self.compute_macro_f1(node_metrics['f1'], edge_strict_metrics['f1'])
        results['macro'] = macro_metrics

        # 5. Graph Edit Distance (optional, for small graphs)
        if len(gold_graph['nodes']) <= 20:
            ged_metrics = self.compute_graph_edit_distance(gold_graph['edges'], pred_graph['edges'])
            results['ged'] = ged_metrics

        return results

    def print_report(self, results: Dict, method_name: str = "Method"):
        """
        Print human-readable evaluation report

        Args:
            results: Evaluation results from evaluate()
            method_name: Name of the method being evaluated
        """
        print("=" * 70)
        print(f"KNOWLEDGE GRAPH EVALUATION: {method_name}")
        print("=" * 70)

        # Macro F1 (Main metric)
        print("\n📊 UNIFIED METRIC (Macro-Averaged F1)")
        print(f"  **Macro F1 Score: {results['macro']['macro_f1']:.4f}** ({results['macro']['macro_f1']:.2%})")

        # Node metrics
        print("\n📍 Component: Node Extraction")
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

        # GED (if available)
        if 'ged' in results and results['ged']['ged'] is not None:
            print("\n📏 Graph Edit Distance")
            print(f"  GED:            {results['ged']['ged']:.0f} operations")
            print(f"  Normalized GED: {results['ged']['normalized_ged']:.4f}")
            print(f"  Similarity:     {results['ged']['similarity']:.4f} ({results['ged']['similarity']:.2%})")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test the evaluator
    gold_graph = {
        'nodes': ['ice thickness', 'temperature', 'salinity', 'albedo', 'wind speed'],
        'edges': [
            {'source': 'temperature', 'relation': 'INCREASES', 'target': 'ice melt'},
            {'source': 'albedo', 'relation': 'CONTROLS', 'target': 'heat flux'},
            {'source': 'wind speed', 'relation': 'INFLUENCES', 'target': 'ocean current'}
        ]
    }

    pred_graph = {
        'nodes': ['ice thickness', 'temperature', 'albedo', 'wind speed', 'precipitation'],
        'edges': [
            {'source': 'temperature', 'relation': 'INCREASES', 'target': 'ice melt'},
            {'source': 'albedo', 'relation': 'AFFECTS', 'target': 'heat flux'},
            {'source': 'wind speed', 'relation': 'INFLUENCES', 'target': 'ice drift'}
        ]
    }

    evaluator = KGEvaluator()
    results = evaluator.evaluate(gold_graph, pred_graph)
    evaluator.print_report(results, "Test Method")
