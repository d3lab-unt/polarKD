#!/usr/bin/env python3
"""
Knowledge Graph Evaluation Script
Runs PolarKD and all baseline methods, generates comparison table
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Import your existing modules
from keywords_extraction import process, text_extraction
from baseline_methods import (
    BaselineCooccurrence,
    BaselineRuleBased,
    BaselineTFIDFOnly,
    BaselineYAKEOnly
)
from kg_evaluation_metrics_relaxed import KGEvaluatorRelaxed as KGEvaluator


def load_gold_standard(gold_standard_path):
    """
    Load manually annotated gold standard

    Args:
        gold_standard_path: Path to gold_standard.json file

    Returns:
        dict: Gold standard graphs for each PDF
    """
    with open(gold_standard_path, 'r') as f:
        gold_standard = json.load(f)

    # Remove metadata
    if '_annotation_instructions' in gold_standard:
        del gold_standard['_annotation_instructions']

    return gold_standard


def run_polarkd(pdf_path, k=15):
    """
    Run PolarKD pipeline on a PDF

    Args:
        pdf_path: Path to PDF file
        k: Number of keywords to extract

    Returns:
        dict: {'nodes': [...], 'edges': [...]}
    """
    print(f"  Running PolarKD on {os.path.basename(pdf_path)}...")

    # Extract text
    text = text_extraction(pdf_path)

    # Run PolarKD pipeline (process function)
    # Returns: (nodes, edges, dataset_info, keywords_metadata)
    # IMPORTANT: filter_variables=False to match gold_standard (uses all keywords, not just variables)
    nodes, edges, dataset_info, keywords_metadata = process(pdf_path, k=k, filter_variables=False)

    return {
        'nodes': list(nodes),
        'edges': edges  # Already in correct format
    }


def run_baseline_methods(text, k=15):
    """
    Run all baseline methods

    Args:
        text: Extracted text from PDF
        k: Number of keywords to extract

    Returns:
        dict: Results from each baseline method
    """
    results = {}

    # Co-occurrence
    print("    Running Co-occurrence baseline...")
    cooccurrence = BaselineCooccurrence()
    results['cooccurrence'] = cooccurrence.process(text, k)

    # Rule-based
    print("    Running Rule-based baseline...")
    rule_based = BaselineRuleBased()
    results['rule_based'] = rule_based.process(text, k)

    # TF-IDF Only
    print("    Running TF-IDF Only baseline...")
    tfidf_only = BaselineTFIDFOnly()
    results['tfidf_only'] = tfidf_only.process(text, k)

    # YAKE Only
    print("    Running YAKE Only baseline...")
    yake_only = BaselineYAKEOnly()
    results['yake_only'] = yake_only.process(text, k)

    return results


def evaluate_all_methods(pdf_folder, gold_standard_path, k=15):
    """
    Evaluate PolarKD and all baselines against gold standard

    Args:
        pdf_folder: Path to folder containing test PDFs
        gold_standard_path: Path to gold_standard.json
        k: Number of keywords to extract

    Returns:
        dict: Evaluation results for all methods
    """
    # Load gold standard
    print("Loading gold standard...")
    gold_standard = load_gold_standard(gold_standard_path)

    # Initialize evaluator with fuzzy matching (80% similarity threshold)
    evaluator = KGEvaluator(node_similarity_threshold=80)

    # Store results for all methods
    all_results = {
        'polarkd': [],
        'cooccurrence': [],
        'rule_based': [],
        'tfidf_only': [],
        'yake_only': []
    }

    # Process each PDF
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        if pdf_file not in gold_standard:
            print(f"Warning: No gold standard for {pdf_file}, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {pdf_file}")
        print(f"{'='*70}")

        pdf_path = os.path.join(pdf_folder, pdf_file)
        gold_graph = gold_standard[pdf_file]

        # Skip if gold standard is empty
        if not gold_graph['nodes'] or not gold_graph['edges']:
            print(f"Warning: Empty gold standard for {pdf_file}, skipping...")
            continue

        # Extract text once for all baselines
        text = text_extraction(pdf_path)

        # 1. Run PolarKD
        try:
            polarkd_graph = run_polarkd(pdf_path, k)
            polarkd_results = evaluator.evaluate(gold_graph, polarkd_graph)
            all_results['polarkd'].append(polarkd_results)
            print(f"  ✓ PolarKD: Macro F1 = {polarkd_results['macro']['macro_f1']:.4f}")
        except Exception as e:
            print(f"  ✗ PolarKD failed: {e}")

        # 2. Run baselines
        baseline_results = run_baseline_methods(text, k)

        # Evaluate Co-occurrence
        try:
            cooccurrence_eval = evaluator.evaluate(gold_graph, baseline_results['cooccurrence'])
            all_results['cooccurrence'].append(cooccurrence_eval)
            print(f"  ✓ Co-occurrence: Macro F1 = {cooccurrence_eval['macro']['macro_f1']:.4f}")
        except Exception as e:
            print(f"  ✗ Co-occurrence failed: {e}")

        # Evaluate Rule-based
        try:
            rule_based_eval = evaluator.evaluate(gold_graph, baseline_results['rule_based'])
            all_results['rule_based'].append(rule_based_eval)
            print(f"  ✓ Rule-based: Macro F1 = {rule_based_eval['macro']['macro_f1']:.4f}")
        except Exception as e:
            print(f"  ✗ Rule-based failed: {e}")

        # Evaluate TF-IDF Only (no edges)
        try:
            tfidf_eval = evaluator.evaluate(gold_graph, baseline_results['tfidf_only'])
            all_results['tfidf_only'].append(tfidf_eval)
            print(f"  ✓ TF-IDF Only: Node F1 = {tfidf_eval['node']['f1']:.4f}")
        except Exception as e:
            print(f"  ✗ TF-IDF Only failed: {e}")

        # Evaluate YAKE Only (no edges)
        try:
            yake_eval = evaluator.evaluate(gold_graph, baseline_results['yake_only'])
            all_results['yake_only'].append(yake_eval)
            print(f"  ✓ YAKE Only: Node F1 = {yake_eval['node']['f1']:.4f}")
        except Exception as e:
            print(f"  ✗ YAKE Only failed: {e}")

    return all_results


def aggregate_results(all_results):
    """
    Aggregate results across all PDFs (compute average metrics)

    Args:
        all_results: Results from evaluate_all_methods()

    Returns:
        dict: Aggregated metrics for each method
    """
    aggregated = {}

    for method, results_list in all_results.items():
        if not results_list:
            aggregated[method] = {
                'macro_f1': None,
                'node_f1': None,
                'edge_f1_strict': None,
                'edge_f1_relaxed': None
            }
            continue

        # Compute averages
        macro_f1_values = [r['macro']['macro_f1'] for r in results_list]
        node_f1_values = [r['node']['f1'] for r in results_list]
        edge_strict_f1_values = [r['edge_strict']['f1'] for r in results_list]
        edge_relaxed_f1_values = [r['edge_relaxed']['f1'] for r in results_list]

        aggregated[method] = {
            'macro_f1': sum(macro_f1_values) / len(macro_f1_values),
            'node_f1': sum(node_f1_values) / len(node_f1_values),
            'edge_f1_strict': sum(edge_strict_f1_values) / len(edge_strict_f1_values),
            'edge_f1_relaxed': sum(edge_relaxed_f1_values) / len(edge_relaxed_f1_values)
        }

    return aggregated


def generate_comparison_table(aggregated_results):
    """
    Generate comparison table (like in your paper)

    Args:
        aggregated_results: Output from aggregate_results()

    Returns:
        pandas.DataFrame: Comparison table
    """
    # Map method names to display names
    method_names = {
        'polarkd': 'PolarKD',
        'cooccurrence': 'Co-occurrence',
        'rule_based': 'Rule-based',
        'tfidf_only': 'TF-IDF Only',
        'yake_only': 'YAKE Only'
    }

    rows = []
    for method_key, display_name in method_names.items():
        metrics = aggregated_results[method_key]

        # Handle N/A for methods without edge extraction
        macro_f1 = f"{metrics['macro_f1']:.2f}" if metrics['macro_f1'] is not None else "N/A"
        node_f1 = f"{metrics['node_f1']:.2f}" if metrics['node_f1'] is not None else "N/A"

        # TF-IDF and YAKE don't extract edges
        if method_key in ['tfidf_only', 'yake_only']:
            edge_strict = "N/A"
            edge_relaxed = "N/A"
        else:
            edge_strict = f"{metrics['edge_f1_strict']:.2f}" if metrics['edge_f1_strict'] is not None else "N/A"
            edge_relaxed = f"{metrics['edge_f1_relaxed']:.2f}" if metrics['edge_f1_relaxed'] is not None else "N/A"

        rows.append({
            'Method': display_name,
            'Macro F1 ⭐': macro_f1,
            'Node F1': node_f1,
            'Edge F1 (Strict)': edge_strict,
            'Edge F1 (Relaxed)': edge_relaxed
        })

    df = pd.DataFrame(rows)
    return df


def main():
    """Main execution"""
    # Configuration
    PDF_FOLDER = os.path.expanduser("~/Downloads/pdf_folder")
    GOLD_STANDARD_PATH = "../gold_standard.json"  # You need to create this manually
    K = 15  # Number of keywords

    print("="*70)
    print("KNOWLEDGE GRAPH EVALUATION - PolarKD vs Baselines")
    print("="*70)

    # Check if gold standard exists
    if not os.path.exists(GOLD_STANDARD_PATH):
        print(f"\n❌ ERROR: Gold standard not found at {GOLD_STANDARD_PATH}")
        print("\nPlease create gold_standard.json by:")
        print("1. Manually annotating nodes and edges for each PDF")
        print("2. Use gold_standard_template.json as a starting point")
        sys.exit(1)

    # Run evaluation
    print(f"\nPDF Folder: {PDF_FOLDER}")
    print(f"Gold Standard: {GOLD_STANDARD_PATH}")
    print(f"Keywords (k): {K}\n")

    all_results = evaluate_all_methods(PDF_FOLDER, GOLD_STANDARD_PATH, k=K)

    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATING RESULTS ACROSS ALL PDFs")
    print("="*70)

    aggregated = aggregate_results(all_results)

    # Generate comparison table
    comparison_table = generate_comparison_table(aggregated)

    # Print table
    print("\n" + "="*70)
    print("FINAL COMPARISON TABLE")
    print("="*70)
    print()
    print(comparison_table.to_string(index=False))
    print()

    # Save to CSV
    output_csv = "kg_evaluation_results.csv"
    comparison_table.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")

    # Print markdown table for paper
    print("\n" + "="*70)
    print("MARKDOWN TABLE (For Paper)")
    print("="*70)
    print()
    print(comparison_table.to_markdown(index=False))
    print()


if __name__ == "__main__":
    main()
