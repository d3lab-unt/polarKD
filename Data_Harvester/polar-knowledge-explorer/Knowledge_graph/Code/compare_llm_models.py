"""
Compare LLM Models for RAG Answer Quality
Evaluates: LLaMA 3, Mistral 7B, Qwen 2.5 7B
Uses RAGAS metrics (Faithfulness + Answer Relevancy)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
import numpy as np

from qa_module import QASystem
from evaluation_metrics import RAGASEvaluator


def generate_test_queries() -> List[str]:
    """Generate test queries for evaluation"""
    return [
        "What datasets are mentioned in the documents?",
        "What are the main findings and results?",
        "What methodologies are used?",
        "What statistical analysis methods are employed?",
        "What are the limitations mentioned?",
        "What future work is proposed?",
        "How is data collected and processed?",
        "What are the key contributions?",
        "What related work is cited?",
        "What evaluation metrics are used?",
    ]


def evaluate_llm_model(model_name: str,
                       pdf_folder: str,
                       test_queries: List[str]) -> Dict[str, Any]:
    """
    Evaluate a single LLM model with the RAG system

    Args:
        model_name: Ollama model name (e.g., 'llama3', 'mistral', 'qwen2.5:7b')
        pdf_folder: Path to PDF folder
        test_queries: List of test questions

    Returns:
        Evaluation results including RAGAS metrics, queries, chunks, and answers
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING LLM: {model_name}")
    print(f"{'='*80}")

    # Initialize QA system with this LLM
    print(f"\n[1/4] Initializing QA system with {model_name}...")
    qa_system = QASystem(model_name=model_name, retrieval_mode='hybrid')

    # Load PDFs
    print(f"\n[2/4] Loading PDFs...")
    pdf_files = list(Path(pdf_folder).rglob("*.pdf"))

    loaded = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"  [{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            qa_system.add_document(pdf_path.name, pdf_path=str(pdf_path))
            loaded += 1
        except Exception as e:
            print(f"    ❌ Error: {e}")

    print(f"✅ Loaded {loaded} PDFs")

    # Collect detailed query-by-query results
    print(f"\n[3/4] Generating answers for all queries...")
    detailed_results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n  [{i}/{len(test_queries)}] Processing: {query[:60]}...")

        try:
            # Get retrieved chunks
            chunks = qa_system.find_relevant_chunks(query, top_k=5, verbose=False)

            # Get answer
            answer = qa_system.answer_question(query, verbose=False)

            # Store detailed information
            detailed_results.append({
                'query_id': i,
                'query': query,
                'num_chunks_retrieved': len(chunks),
                'retrieved_chunks': [
                    {
                        'chunk_id': j + 1,
                        'filename': chunk['filename'],
                        'score': float(chunk['score']),
                        'retrieval_method': chunk.get('retrieval_method', 'unknown'),
                        'content': chunk['chunk']
                    }
                    for j, chunk in enumerate(chunks)
                ],
                'answer': answer,
                'answer_length_chars': len(answer),
                'answer_length_tokens': qa_system.count_tokens(answer) if answer else 0
            })

            print(f"    ✓ Retrieved {len(chunks)} chunks, generated answer ({len(answer)} chars)")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            detailed_results.append({
                'query_id': i,
                'query': query,
                'error': str(e)
            })

    # Evaluate with RAGAS
    print(f"\n[4/4] Evaluating with RAGAS metrics...")
    evaluator = RAGASEvaluator(qa_system, model_name=model_name)

    ragas_results = evaluator.evaluate_test_set(test_queries, top_k=5)

    # Merge RAGAS scores into detailed results
    if 'results' in ragas_results:
        for i, ragas_item in enumerate(ragas_results['results']):
            if i < len(detailed_results) and 'error' not in detailed_results[i]:
                detailed_results[i]['faithfulness_score'] = ragas_item.get('faithfulness', 0.0)
                detailed_results[i]['relevancy_score'] = ragas_item.get('relevancy', 0.0)

    # Collect final results
    result = {
        'model_name': model_name,
        'num_pdfs': loaded,
        'num_queries': len(test_queries),
        'ragas_metrics': {
            'avg_faithfulness': ragas_results['avg_faithfulness'],
            'avg_relevancy': ragas_results['avg_relevancy'],
            'combined_score': ragas_results['combined_score']
        },
        'detailed_query_results': detailed_results
    }

    print(f"\n{'='*80}")
    print(f"RESULTS FOR {model_name}")
    print(f"{'='*80}")
    print(f"Avg Faithfulness: {ragas_results['avg_faithfulness']:.4f}")
    print(f"Avg Relevancy:    {ragas_results['avg_relevancy']:.4f}")
    print(f"Combined Score:   {ragas_results['combined_score']:.4f}")
    print(f"{'='*80}\n")

    return result


def compare_llm_models(pdf_folder: str, output_dir: str = "llm_comparison"):
    """
    Compare multiple LLM models for RAG answer quality

    Args:
        pdf_folder: Path to PDF folder
        output_dir: Output directory for results
    """
    print("="*80)
    print("LLM MODEL COMPARISON FOR RAG SYSTEM")
    print("="*80)
    print("\nModels to evaluate:")
    print("  1. LLaMA 3 (8B) - Current baseline")
    print("  2. Mistral 7B - Instruction-following specialist")
    print("  3. Qwen 2.5 (7B) - Reasoning specialist")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Generate test queries
    print("\n[1/4] Generating test queries...")
    test_queries = generate_test_queries()
    print(f"✅ Generated {len(test_queries)} test queries")

    # Models to evaluate
    models = [
        "llama3",      # LLaMA 3 8B (current)
        "mistral",     # Mistral 7B
        "qwen2.5:7b"   # Qwen 2.5 7B
    ]

    all_results = {}

    # Evaluate each model
    for i, model_name in enumerate(models, 2):
        print(f"\n[{i}/4] Evaluating {model_name}...")

        try:
            result = evaluate_llm_model(model_name, pdf_folder, test_queries)
            all_results[model_name] = result
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {
                'error': str(e),
                'model_name': model_name
            }

    # Generate comparison report
    print(f"\n[4/4] Generating comparison report...")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': datetime.now().isoformat(),
        'pdf_folder': pdf_folder,
        'test_queries': test_queries,
        'models_evaluated': models,
        'results': all_results
    }

    json_file = os.path.join(output_dir, f"llm_comparison_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Results saved to: {json_file}")

    # Generate comparison table
    print("\n" + "="*80)
    print("LLM MODEL COMPARISON TABLE")
    print("="*80)

    print(f"\n{'Model':<20} {'Faithfulness':<15} {'Relevancy':<15} {'Combined':<15}")
    print("-"*65)

    for model_name, result in all_results.items():
        if 'error' in result:
            print(f"{model_name:<20} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")
        else:
            metrics = result['ragas_metrics']
            faithfulness = metrics['avg_faithfulness']
            relevancy = metrics['avg_relevancy']
            combined = metrics['combined_score']

            print(f"{model_name:<20} {faithfulness:<15.4f} {relevancy:<15.4f} {combined:<15.4f}")

    # Find best model
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}

    if valid_results:
        # Best faithfulness
        best_faith = max(valid_results.items(),
                        key=lambda x: x[1]['ragas_metrics']['avg_faithfulness'])
        print(f"\n✅ Best Faithfulness: {best_faith[0]} "
              f"({best_faith[1]['ragas_metrics']['avg_faithfulness']:.4f})")

        # Best relevancy
        best_rel = max(valid_results.items(),
                      key=lambda x: x[1]['ragas_metrics']['avg_relevancy'])
        print(f"✅ Best Relevancy: {best_rel[0]} "
              f"({best_rel[1]['ragas_metrics']['avg_relevancy']:.4f})")

        # Best combined
        best_combined = max(valid_results.items(),
                           key=lambda x: x[1]['ragas_metrics']['combined_score'])
        print(f"✅ Best Overall: {best_combined[0]} "
              f"({best_combined[1]['ragas_metrics']['combined_score']:.4f})")

        # Analysis
        print("\n" + "-"*80)
        print("ANALYSIS")
        print("-"*80)

        llama_score = all_results.get('llama3', {}).get('ragas_metrics', {}).get('combined_score', 0)

        for model_name, result in valid_results.items():
            if model_name != 'llama3' and 'ragas_metrics' in result:
                model_score = result['ragas_metrics']['combined_score']
                diff = ((model_score - llama_score) / llama_score * 100) if llama_score > 0 else 0

                if diff > 0:
                    print(f"\n{model_name} performs {diff:.1f}% BETTER than LLaMA 3")
                elif diff < 0:
                    print(f"\n{model_name} performs {abs(diff):.1f}% WORSE than LLaMA 3")
                else:
                    print(f"\n{model_name} performs EQUALLY to LLaMA 3")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    return output


def main():
    """Main function"""

    # Configuration - UPDATE THIS PATH ON YOUR SERVER
    PDF_FOLDER = "/home/ad1457@students.ad.unt.edu/Downloads/pdf_folder"
    OUTPUT_DIR = "llm_comparison"

    # Check if custom path provided
    if len(sys.argv) > 1:
        PDF_FOLDER = sys.argv[1]

    # Run comparison
    compare_llm_models(PDF_FOLDER, OUTPUT_DIR)


if __name__ == "__main__":
    main()
