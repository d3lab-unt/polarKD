#!/usr/bin/env python3
"""
Graph Edit Distance (GED) Evaluation for Knowledge Graphs
Complements F1-based metrics with structural similarity measure
"""

import networkx as nx
import json
from typing import Dict, List, Tuple
import numpy as np


class GEDEvaluator:
    """
    Evaluates knowledge graphs using Graph Edit Distance

    GED measures minimum edit operations (add/delete/substitute nodes/edges)
    needed to transform predicted graph into gold standard graph.

    Lower GED = Better (more similar to gold standard)
    """

    def __init__(self):
        pass

    def build_graph(self, kg_data: Dict) -> nx.DiGraph:
        """
        Convert knowledge graph dict to NetworkX directed graph

        Args:
            kg_data: {'nodes': [...], 'edges': [{'source': ..., 'relation': ..., 'target': ...}]}

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes
        for node in kg_data['nodes']:
            G.add_node(node.lower())

        # Add edges with relation labels
        for edge in kg_data['edges']:
            source = edge['source'].lower()
            target = edge['target'].lower()
            relation = edge['relation'].upper()

            # Add edge with relation as attribute
            G.add_edge(source, target, relation=relation)

        return G

    def compute_ged(self, gold_graph: nx.DiGraph, pred_graph: nx.DiGraph,
                    use_edge_labels: bool = True) -> float:
        """
        Compute Graph Edit Distance between two graphs

        Args:
            gold_graph: Gold standard NetworkX graph
            pred_graph: Predicted NetworkX graph
            use_edge_labels: If True, edge labels (relations) must match

        Returns:
            GED score (lower = better, 0 = identical graphs)
        """
        # Define edit costs
        def node_subst_cost(n1, n2):
            """Cost of substituting node n1 with n2"""
            if n1['label'].lower() == n2['label'].lower():
                return 0
            return 1

        def node_del_cost(n):
            """Cost of deleting a node"""
            return 1

        def node_ins_cost(n):
            """Cost of inserting a node"""
            return 1

        def edge_subst_cost(e1, e2):
            """Cost of substituting edge e1 with e2"""
            if use_edge_labels:
                # Check if relations match
                rel1 = e1.get('relation', '')
                rel2 = e2.get('relation', '')
                if rel1 == rel2:
                    return 0
                return 1
            else:
                # Only check if edge exists (ignore relation type)
                return 0

        def edge_del_cost(e):
            """Cost of deleting an edge"""
            return 1

        def edge_ins_cost(e):
            """Cost of inserting an edge"""
            return 1

        # Add node labels for comparison
        for node in gold_graph.nodes():
            gold_graph.nodes[node]['label'] = node
        for node in pred_graph.nodes():
            pred_graph.nodes[node]['label'] = node

        # Compute GED using NetworkX
        try:
            # Use optimize_graph_edit_distance for better performance
            ged_paths = nx.optimize_graph_edit_distance(
                gold_graph, pred_graph,
                node_subst_cost=node_subst_cost,
                node_del_cost=node_del_cost,
                node_ins_cost=node_ins_cost,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=edge_del_cost,
                edge_ins_cost=edge_ins_cost
            )

            # Get minimum GED
            for ged in ged_paths:
                return ged

            # If no path found, compute simple estimate
            return self.estimate_ged(gold_graph, pred_graph)

        except:
            # Fallback to simple estimate
            return self.estimate_ged(gold_graph, pred_graph)

    def estimate_ged(self, gold_graph: nx.DiGraph, pred_graph: nx.DiGraph) -> float:
        """
        Fast estimate of GED when exact computation fails

        Approximation: GED ≈ node_edits + edge_edits
        """
        gold_nodes = set(gold_graph.nodes())
        pred_nodes = set(pred_graph.nodes())

        gold_edges = set((u, v) for u, v in gold_graph.edges())
        pred_edges = set((u, v) for u, v in pred_graph.edges())

        # Node operations
        node_insertions = len(pred_nodes - gold_nodes)
        node_deletions = len(gold_nodes - pred_nodes)

        # Edge operations
        edge_insertions = len(pred_edges - gold_edges)
        edge_deletions = len(gold_edges - pred_edges)

        ged_estimate = node_insertions + node_deletions + edge_insertions + edge_deletions

        return ged_estimate

    def normalize_ged(self, ged: float, gold_graph: nx.DiGraph) -> float:
        """
        Normalize GED by maximum possible distance

        Returns value in [0, 1] where 0 = identical, 1 = completely different
        """
        # Maximum GED = deleting entire gold graph and inserting predicted graph
        max_ged = gold_graph.number_of_nodes() + gold_graph.number_of_edges()

        if max_ged == 0:
            return 0.0

        return min(1.0, ged / max_ged)

    def compute_ged_similarity(self, ged: float, gold_graph: nx.DiGraph) -> float:
        """
        Convert GED to similarity score in [0, 1]

        Returns: 1 - normalized_GED (1 = identical, 0 = completely different)
        """
        norm_ged = self.normalize_ged(ged, gold_graph)
        return 1.0 - norm_ged

    def evaluate(self, gold_kg: Dict, pred_kg: Dict) -> Dict:
        """
        Complete GED evaluation

        Args:
            gold_kg: Gold standard KG {'nodes': [...], 'edges': [...]}
            pred_kg: Predicted KG {'nodes': [...], 'edges': [...]}

        Returns:
            dict with GED metrics
        """
        # Build graphs
        gold_graph = self.build_graph(gold_kg)
        pred_graph = self.build_graph(pred_kg)

        # Compute GED (strict - with edge labels)
        ged_strict = self.compute_ged(gold_graph, pred_graph, use_edge_labels=True)

        # Compute GED (relaxed - without edge labels)
        ged_relaxed = self.compute_ged(gold_graph, pred_graph, use_edge_labels=False)

        # Normalize and compute similarities
        norm_ged_strict = self.normalize_ged(ged_strict, gold_graph)
        norm_ged_relaxed = self.normalize_ged(ged_relaxed, gold_graph)

        similarity_strict = self.compute_ged_similarity(ged_strict, gold_graph)
        similarity_relaxed = self.compute_ged_similarity(ged_relaxed, gold_graph)

        return {
            'ged_strict': ged_strict,
            'ged_relaxed': ged_relaxed,
            'ged_normalized_strict': norm_ged_strict,
            'ged_normalized_relaxed': norm_ged_relaxed,
            'ged_similarity_strict': similarity_strict,
            'ged_similarity_relaxed': similarity_relaxed,
            'gold_nodes': gold_graph.number_of_nodes(),
            'gold_edges': gold_graph.number_of_edges(),
            'pred_nodes': pred_graph.number_of_nodes(),
            'pred_edges': pred_graph.number_of_edges()
        }


def main():
    """Test GED evaluator"""

    # Example gold standard
    gold_kg = {
        'nodes': ['climate change', 'global warming', 'sea ice'],
        'edges': [
            {'source': 'global warming', 'relation': 'CAUSES', 'target': 'climate change'},
            {'source': 'climate change', 'relation': 'DECREASES', 'target': 'sea ice'}
        ]
    }

    # Example prediction (similar but not identical)
    pred_kg = {
        'nodes': ['climate change', 'warming', 'sea ice', 'temperature'],
        'edges': [
            {'source': 'warming', 'relation': 'RELATES_TO', 'target': 'climate change'},
            {'source': 'climate change', 'relation': 'AFFECTS', 'target': 'sea ice'},
            {'source': 'temperature', 'relation': 'INCREASES', 'target': 'warming'}
        ]
    }

    evaluator = GEDEvaluator()
    results = evaluator.evaluate(gold_kg, pred_kg)

    print("="*70)
    print("GRAPH EDIT DISTANCE EVALUATION")
    print("="*70)
    print(f"\nGold Standard: {results['gold_nodes']} nodes, {results['gold_edges']} edges")
    print(f"Predicted:     {results['pred_nodes']} nodes, {results['pred_edges']} edges")
    print(f"\nGED (Strict):     {results['ged_strict']:.0f} edits")
    print(f"GED (Relaxed):    {results['ged_relaxed']:.0f} edits")
    print(f"\nNormalized GED (Strict):  {results['ged_normalized_strict']:.3f}")
    print(f"Normalized GED (Relaxed): {results['ged_normalized_relaxed']:.3f}")
    print(f"\nGED Similarity (Strict):  {results['ged_similarity_strict']:.3f} (1=identical)")
    print(f"GED Similarity (Relaxed): {results['ged_similarity_relaxed']:.3f} (1=identical)")
    print("\nInterpretation:")
    print("  - GED: Number of edit operations needed (lower = better)")
    print("  - Normalized GED: GED / max_possible_GED (0-1, lower = better)")
    print("  - GED Similarity: 1 - Normalized GED (0-1, higher = better)")


if __name__ == "__main__":
    main()
