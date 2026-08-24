"""
Comparison Script: Hybrid vs FAISS-only vs BM25-only Retrieval Systems
Generates comprehensive evaluation comparing different retrieval approaches
Author: Research Evaluation Module
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Import required modules
from qa_module import QASystem
from evaluation_metrics import (
    MRRCalculator,
    NDCGCalculator,
    RecallCalculator,
    MMRCalculator
)


def find_pdf_files(folder_path: str) -> List[str]:
    """Find all PDF files in a folder recursively"""
    pdf_files = []
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Error: Folder '{folder_path}' does not exist")
        return []

    for pdf_path in folder.rglob("*.pdf"):
        pdf_files.append(str(pdf_path))

    return sorted(pdf_files)


def generate_test_dataset() -> List[Dict[str, Any]]:
    """
    Generate test dataset with queries and relevant chunk identifiers

    Returns:
        List of test cases with query and relevant_chunks
    """
    test_set = [
        {
            'query': 'What datasets are mentioned in the documents?',
            'relevant_chunks': ['dataset', 'data', 'ERA5', 'CMIP6', 'reanalysis'],
            'query_type': 'factual'
        },
        {
            'query': 'What are the main findings and results?',
            'relevant_chunks': ['findings', 'results', 'conclusion', 'showed', 'demonstrated'],
            'query_type': 'analytical'
        },
        {
            'query': 'What methodologies are used?',
            'relevant_chunks': ['method', 'approach', 'technique', 'algorithm', 'model'],
            'query_type': 'methodological'
        },
        {
            'query': 'What statistical analysis methods are employed?',
            'relevant_chunks': ['statistical', 'analysis', 'regression', 'correlation', 'significance'],
            'query_type': 'specific'
        },
        {
            'query': 'What are the limitations mentioned?',
            'relevant_chunks': ['limitation', 'constraint', 'challenge', 'issue', 'problem'],
            'query_type': 'critical'
        },
        {
            'query': 'What future work is proposed?',
            'relevant_chunks': ['future', 'further', 'next', 'upcoming', 'planned'],
            'query_type': 'forward-looking'
        },
        {
            'query': 'How is data collected and processed?',
            'relevant_chunks': ['collection', 'processing', 'preprocessing', 'pipeline', 'workflow'],
            'query_type': 'procedural'
        },
        {
            'query': 'What are the key contributions?',
            'relevant_chunks': ['contribution', 'novel', 'innovative', 'propose', 'introduce'],
            'query_type': 'impact'
        },
        {
            'query': 'What related work is cited?',
            'relevant_chunks': ['related', 'prior', 'previous', 'literature', 'cited'],
            'query_type': 'contextual'
        },
        {
            'query': 'What evaluation metrics are used?',
            'relevant_chunks': ['metric', 'evaluation', 'performance', 'accuracy', 'precision'],
            'query_type': 'methodological'
        },
    ]

    return test_set


def load_pdfs_into_system(qa_system: QASystem, pdf_files: List[str]) -> tuple:
    """
    Load PDFs into QA system

    Returns:
        Tuple of (loaded_count, failed_pdfs)
    """
    loaded_count = 0
    failed_pdfs = []

    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_name = Path(pdf_path).name
        print(f"  [{i}/{len(pdf_files)}] Loading: {pdf_name}")

        try:
            qa_system.add_document(pdf_name, pdf_path=pdf_path)
            loaded_count += 1
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            failed_pdfs.append((pdf_name, str(e)))

    return loaded_count, failed_pdfs


def evaluate_system(system_name: str,
                   qa_system: QASystem,
                   test_set: List[Dict[str, Any]],
                   k_values: List[int] = [5, 10]) -> Dict[str, Any]:
    """
    Evaluate a single retrieval system with all metrics

    Args:
        system_name: Name of the system (e.g., "Hybrid", "FAISS-only")
        qa_system: QA system instance
        test_set: Test dataset
        k_values: K values for evaluation

    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {system_name}")
    print(f"{'='*80}")

    # Initialize calculators
    mrr_calc = MRRCalculator()
    ndcg_calc = NDCGCalculator()
    recall_calc = RecallCalculator()
    mmr_calc = MMRCalculator(lambda_param=0.5)

    # Evaluate each query
    for i, test_case in enumerate(test_set, 1):
        query = test_case['query']
        relevant_chunks = test_case['relevant_chunks']

        print(f"\n[{i}/{len(test_set)}] {query[:60]}...")

        try:
            # MRR evaluation
            mrr_calc.evaluate_query(qa_system, query, relevant_chunks, top_k=max(k_values))

            # NDCG evaluation
            ndcg_calc.evaluate_query(qa_system, query, relevant_chunks, k_values=k_values)

            # Recall evaluation
            recall_calc.evaluate_query(qa_system, query, relevant_chunks, k_values=k_values)

            # MMR (diversity) evaluation
            mmr_calc.evaluate_query_diversity(qa_system, query, top_k=max(k_values))

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Collect statistics
    mrr_stats = mrr_calc.get_statistics()
    ndcg_stats = ndcg_calc.get_statistics(k_values)
    recall_stats = recall_calc.get_statistics(k_values)
    mmr_stats = mmr_calc.get_statistics()

    # Print reports
    mrr_calc.print_report()
    ndcg_calc.print_report(k_values)
    recall_calc.print_report(k_values)
    mmr_calc.print_report()

    return {
        'system_name': system_name,
        'mrr': mrr_stats,
        'ndcg': ndcg_stats,
        'recall': recall_stats,
        'mmr': mmr_stats,
        'detailed_results': {
            'mrr': mrr_calc.detailed_results,
            'ndcg': ndcg_calc.detailed_results,
            'recall': recall_calc.detailed_results,
            'mmr': mmr_calc.detailed_results
        }
    }


def run_comparison_evaluation(pdf_folder: str,
                              output_dir: str = "comparison_results",
                              k_values: List[int] = [5, 10]):
    """
    Run comprehensive comparison of all retrieval systems

    Args:
        pdf_folder: Path to folder containing PDFs
        output_dir: Directory to save results
        k_values: K values for evaluation metrics
    """
    print("="*80)
    print("RETRIEVAL SYSTEM COMPARISON EVALUATION")
    print("="*80)
    print(f"\nPDF Folder: {pdf_folder}")
    print(f"Output Directory: {output_dir}")
    print(f"K Values: {k_values}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Find PDF files
    print("\n" + "-"*80)
    print("STEP 1: Finding PDF files...")
    print("-"*80)
    pdf_files = find_pdf_files(pdf_folder)

    if not pdf_files:
        print("❌ No PDF files found")
        return

    print(f"✅ Found {len(pdf_files)} PDF files")

    # Step 2: Generate test dataset
    print("\n" + "-"*80)
    print("STEP 2: Generating test dataset...")
    print("-"*80)
    test_set = generate_test_dataset()
    print(f"✅ Generated {len(test_set)} test queries")

    # Step 3: Evaluate FAISS-only System (Dense Retrieval)
    print("\n" + "="*80)
    print("STEP 3: Evaluating FAISS-ONLY System (Dense Retrieval)")
    print("="*80)

    qa_faiss = QASystem(retrieval_mode='faiss')
    print("Loading PDFs into FAISS-only system...")
    loaded_count_faiss, failed_faiss = load_pdfs_into_system(qa_faiss, pdf_files)
    print(f"✅ Loaded {loaded_count_faiss}/{len(pdf_files)} PDFs")

    faiss_results = evaluate_system("FAISS-only (Dense)", qa_faiss, test_set, k_values)

    # Step 4: Evaluate BM25-only System (Sparse Retrieval)
    print("\n" + "="*80)
    print("STEP 4: Evaluating BM25-ONLY System (Sparse Retrieval)")
    print("="*80)

    qa_bm25 = QASystem(retrieval_mode='bm25')
    print("Loading PDFs into BM25-only system...")
    loaded_count_bm25, failed_bm25 = load_pdfs_into_system(qa_bm25, pdf_files)
    print(f"✅ Loaded {loaded_count_bm25}/{len(pdf_files)} PDFs")

    bm25_results = evaluate_system("BM25-only (Sparse)", qa_bm25, test_set, k_values)

    # Step 5: Evaluate Hybrid System (FAISS + BM25)
    print("\n" + "="*80)
    print("STEP 5: Evaluating HYBRID System (FAISS + BM25)")
    print("="*80)

    qa_hybrid = QASystem(retrieval_mode='hybrid')
    print("Loading PDFs into Hybrid system...")
    loaded_count_hybrid, failed_hybrid = load_pdfs_into_system(qa_hybrid, pdf_files)
    print(f"✅ Loaded {loaded_count_hybrid}/{len(pdf_files)} PDFs")

    hybrid_results = evaluate_system("Hybrid (FAISS + BM25)", qa_hybrid, test_set, k_values)

    # Step 6: Generate comparison report
    print("\n" + "="*80)
    print("STEP 6: Generating Comparison Report")
    print("="*80)

    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'pdf_folder': pdf_folder,
        'num_pdfs': len(pdf_files),
        'num_queries': len(test_set),
        'k_values': k_values,
        'systems': {
            'faiss_only': faiss_results,
            'bm25_only': bm25_results,
            'hybrid': hybrid_results
        }
    }

    # Save detailed JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python native types"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    comparison_results_serializable = convert_to_serializable(comparison_results)

    with open(json_file, 'w') as f:
        json.dump(comparison_results_serializable, f, indent=2)

    print(f"✅ Detailed results saved to: {json_file}")

    # Generate comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE (3-WAY)")
    print("="*80)

    # Extract key metrics for comparison
    print(f"\n{'Metric':<20} {'FAISS':<12} {'BM25':<12} {'Hybrid':<12} {'Best':<10}")
    print("-"*66)

    # MRR comparison
    mrr_faiss = faiss_results['mrr'].get('mrr', 0.0)
    mrr_bm25 = bm25_results['mrr'].get('mrr', 0.0)
    mrr_hybrid = hybrid_results['mrr'].get('mrr', 0.0)
    best_mrr = max(mrr_faiss, mrr_bm25, mrr_hybrid)
    print(f"{'MRR':<20} {mrr_faiss:<12.4f} {mrr_bm25:<12.4f} {mrr_hybrid:<12.4f} {best_mrr:<10.4f}")

    # NDCG@5 comparison
    ndcg5_faiss = faiss_results['ndcg'].get('avg_ndcg@5', 0.0)
    ndcg5_bm25 = bm25_results['ndcg'].get('avg_ndcg@5', 0.0)
    ndcg5_hybrid = hybrid_results['ndcg'].get('avg_ndcg@5', 0.0)
    best_ndcg5 = max(ndcg5_faiss, ndcg5_bm25, ndcg5_hybrid)
    print(f"{'NDCG@5':<20} {ndcg5_faiss:<12.4f} {ndcg5_bm25:<12.4f} {ndcg5_hybrid:<12.4f} {best_ndcg5:<10.4f}")

    # NDCG@10 comparison
    ndcg10_faiss = faiss_results['ndcg'].get('avg_ndcg@10', 0.0)
    ndcg10_bm25 = bm25_results['ndcg'].get('avg_ndcg@10', 0.0)
    ndcg10_hybrid = hybrid_results['ndcg'].get('avg_ndcg@10', 0.0)
    best_ndcg10 = max(ndcg10_faiss, ndcg10_bm25, ndcg10_hybrid)
    print(f"{'NDCG@10':<20} {ndcg10_faiss:<12.4f} {ndcg10_bm25:<12.4f} {ndcg10_hybrid:<12.4f} {best_ndcg10:<10.4f}")

    # Recall@5 comparison
    recall5_faiss = faiss_results['recall'].get('avg_recall@5', 0.0)
    recall5_bm25 = bm25_results['recall'].get('avg_recall@5', 0.0)
    recall5_hybrid = hybrid_results['recall'].get('avg_recall@5', 0.0)
    best_recall5 = max(recall5_faiss, recall5_bm25, recall5_hybrid)
    print(f"{'Recall@5':<20} {recall5_faiss:<12.4f} {recall5_bm25:<12.4f} {recall5_hybrid:<12.4f} {best_recall5:<10.4f}")

    # Recall@10 comparison
    recall10_faiss = faiss_results['recall'].get('avg_recall@10', 0.0)
    recall10_bm25 = bm25_results['recall'].get('avg_recall@10', 0.0)
    recall10_hybrid = hybrid_results['recall'].get('avg_recall@10', 0.0)
    best_recall10 = max(recall10_faiss, recall10_bm25, recall10_hybrid)
    print(f"{'Recall@10':<20} {recall10_faiss:<12.4f} {recall10_bm25:<12.4f} {recall10_hybrid:<12.4f} {best_recall10:<10.4f}")

    # Hit Rate comparison
    hit_faiss = faiss_results['recall'].get('hit_rate', 0.0)
    hit_bm25 = bm25_results['recall'].get('hit_rate', 0.0)
    hit_hybrid = hybrid_results['recall'].get('hit_rate', 0.0)
    best_hit = max(hit_faiss, hit_bm25, hit_hybrid)
    print(f"{'Hit Rate':<20} {hit_faiss:<12.4f} {hit_bm25:<12.4f} {hit_hybrid:<12.4f} {best_hit:<10.4f}")

    # Diversity comparison
    div_faiss = faiss_results['mmr'].get('avg_diversity', 0.0)
    div_bm25 = bm25_results['mmr'].get('avg_diversity', 0.0)
    div_hybrid = hybrid_results['mmr'].get('avg_diversity', 0.0)
    best_div = max(div_faiss, div_bm25, div_hybrid)
    print(f"{'Diversity':<20} {div_faiss:<12.4f} {div_bm25:<12.4f} {div_hybrid:<12.4f} {best_div:<10.4f}")

    print("-"*66)

    # Save comparison table
    table_file = os.path.join(output_dir, f"comparison_table_{timestamp}.txt")
    with open(table_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RETRIEVAL SYSTEM 3-WAY COMPARISON TABLE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PDF Folder: {pdf_folder}\n")
        f.write(f"Number of PDFs: {len(pdf_files)}\n")
        f.write(f"Number of Queries: {len(test_set)}\n\n")

        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<20} {'FAISS':<12} {'BM25':<12} {'Hybrid':<12} {'Best':<10}\n")
        f.write("-"*66 + "\n")
        f.write(f"{'MRR':<20} {mrr_faiss:<12.4f} {mrr_bm25:<12.4f} {mrr_hybrid:<12.4f} {best_mrr:<10.4f}\n")
        f.write(f"{'NDCG@5':<20} {ndcg5_faiss:<12.4f} {ndcg5_bm25:<12.4f} {ndcg5_hybrid:<12.4f} {best_ndcg5:<10.4f}\n")
        f.write(f"{'NDCG@10':<20} {ndcg10_faiss:<12.4f} {ndcg10_bm25:<12.4f} {ndcg10_hybrid:<12.4f} {best_ndcg10:<10.4f}\n")
        f.write(f"{'Recall@5':<20} {recall5_faiss:<12.4f} {recall5_bm25:<12.4f} {recall5_hybrid:<12.4f} {best_recall5:<10.4f}\n")
        f.write(f"{'Recall@10':<20} {recall10_faiss:<12.4f} {recall10_bm25:<12.4f} {recall10_hybrid:<12.4f} {best_recall10:<10.4f}\n")
        f.write(f"{'Hit Rate':<20} {hit_faiss:<12.4f} {hit_bm25:<12.4f} {hit_hybrid:<12.4f} {best_hit:<10.4f}\n")
        f.write(f"{'Diversity':<20} {div_faiss:<12.4f} {div_bm25:<12.4f} {div_hybrid:<12.4f} {best_div:<10.4f}\n")
        f.write("-"*66 + "\n\n")

        # Calculate improvements over baselines
        mrr_hybrid_vs_faiss = ((mrr_hybrid - mrr_faiss) / mrr_faiss * 100) if mrr_faiss > 0 else 0.0
        mrr_hybrid_vs_bm25 = ((mrr_hybrid - mrr_bm25) / mrr_bm25 * 100) if mrr_bm25 > 0 else 0.0
        ndcg10_hybrid_vs_faiss = ((ndcg10_hybrid - ndcg10_faiss) / ndcg10_faiss * 100) if ndcg10_faiss > 0 else 0.0
        ndcg10_hybrid_vs_bm25 = ((ndcg10_hybrid - ndcg10_bm25) / ndcg10_bm25 * 100) if ndcg10_bm25 > 0 else 0.0

        f.write("Key Findings:\n")
        f.write(f"- Hybrid MRR: {mrr_hybrid_vs_faiss:+.1f}% vs FAISS, {mrr_hybrid_vs_bm25:+.1f}% vs BM25\n")
        f.write(f"- Hybrid NDCG@10: {ndcg10_hybrid_vs_faiss:+.1f}% vs FAISS, {ndcg10_hybrid_vs_bm25:+.1f}% vs BM25\n")
        f.write(f"- Best overall: {'Hybrid' if best_mrr == mrr_hybrid else 'FAISS' if best_mrr == mrr_faiss else 'BM25'}\n")

    print(f"\n✅ Comparison table saved to: {table_file}")

    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n📊 Results Directory: {output_dir}/")
    print(f"   - JSON results: {Path(json_file).name}")
    print(f"   - Comparison table: {Path(table_file).name}")

    print("\n" + "="*80)


def main():
    """Main function to run comparison evaluation"""
    # Default configuration
    DEFAULT_PDF_FOLDER = "/home/ad1457@students.ad.unt.edu/Downloads/pdf_folder"
    DEFAULT_OUTPUT_DIR = "comparison_results"
    DEFAULT_K_VALUES = [5, 10]

    # Check command line arguments
    if len(sys.argv) > 1:
        pdf_folder = sys.argv[1]
    else:
        pdf_folder = DEFAULT_PDF_FOLDER

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    # Print usage information
    if len(sys.argv) == 1:
        print("\nUsing default configuration:")
        print(f"  PDF Folder: {pdf_folder}")
        print(f"  Output Directory: {output_dir}")
        print(f"  K Values: {DEFAULT_K_VALUES}")
        print("\nUsage: python compare_retrieval_systems.py [pdf_folder] [output_dir]")
        print("  pdf_folder: Path to folder containing PDFs")
        print("  output_dir: Directory to save results (default: comparison_results)")
        print("\nExample:")
        print("  python compare_retrieval_systems.py /path/to/pdfs results")
        print("")

    # Run comparison
    run_comparison_evaluation(
        pdf_folder=pdf_folder,
        output_dir=output_dir,
        k_values=DEFAULT_K_VALUES
    )


if __name__ == "__main__":
    main()