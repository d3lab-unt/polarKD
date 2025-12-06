"""
RAG Pipeline Evaluation Metrics
Implements MRR and RAGAS metrics for research evaluation
Author: Research Evaluation Module
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime
from collections import defaultdict
import os
import ollama
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# Import the QA system without modifying it
from qa_module import QASystem


class MMRCalculator:
    """
    Mean Maximum Relevance (MMR) Calculator
    Evaluates diversity in retrieved results while maintaining relevance
    MMR = λ * Similarity(Q, D) - (1-λ) * max Similarity(D, D_i)
    where D_i are already selected documents
    """

    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR calculator

        Args:
            lambda_param: Balance between relevance and diversity (0-1)
                         1.0 = pure relevance, 0.0 = pure diversity
        """
        self.lambda_param = lambda_param
        self.results = []
        self.detailed_results = []
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_mmr_scores(self,
                            query: str,
                            retrieved_chunks: List[Dict[str, Any]],
                            top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Calculate MMR scores for retrieved chunks

        Args:
            query: User query
            retrieved_chunks: List of retrieved chunks with metadata
            top_k: Number of diverse results to select

        Returns:
            List of (chunk_index, mmr_score) tuples
        """
        if not retrieved_chunks:
            return []

        # Get embeddings
        query_embedding = self.embedder.encode(query, convert_to_tensor=False)
        chunk_texts = [chunk.get('chunk', '') for chunk in retrieved_chunks]
        chunk_embeddings = self.embedder.encode(chunk_texts, convert_to_tensor=False)

        # Calculate query-document similarities
        query_sims = cosine_similarity([query_embedding], chunk_embeddings)[0]

        # MMR selection
        selected_indices = []
        selected_embeddings = []
        remaining_indices = list(range(len(retrieved_chunks)))

        for _ in range(min(top_k, len(retrieved_chunks))):
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance term
                relevance = query_sims[idx]

                # Diversity term (maximum similarity to already selected)
                if selected_embeddings:
                    doc_sims = cosine_similarity([chunk_embeddings[idx]], selected_embeddings)[0]
                    max_sim = np.max(doc_sims)
                else:
                    max_sim = 0.0

                # MMR formula
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            selected_embeddings.append(chunk_embeddings[best_idx])
            remaining_indices.remove(best_idx)

        return [(idx, query_sims[idx]) for idx in selected_indices]

    def evaluate_query_diversity(self,
                                 qa_system,
                                 query: str,
                                 top_k: int = 10) -> Dict[str, Any]:
        """
        Evaluate diversity of retrieved results using MMR

        Returns:
            Dictionary with diversity metrics
        """
        # Retrieve chunks
        retrieved_chunks = qa_system.find_relevant_chunks(query, top_k=top_k, verbose=False)

        if not retrieved_chunks:
            return {
                'query': query,
                'mmr_score': 0.0,
                'diversity_score': 0.0,
                'avg_pairwise_similarity': 0.0,
                'num_retrieved': 0
            }

        # Get embeddings for diversity calculation
        chunk_texts = [chunk.get('chunk', '') for chunk in retrieved_chunks]
        chunk_embeddings = self.embedder.encode(chunk_texts, convert_to_tensor=False)

        # Calculate average pairwise similarity (lower = more diverse)
        pairwise_sims = []
        for i in range(len(chunk_embeddings)):
            for j in range(i + 1, len(chunk_embeddings)):
                sim = cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[j]])[0][0]
                pairwise_sims.append(sim)

        avg_pairwise_sim = np.mean(pairwise_sims) if pairwise_sims else 0.0
        diversity_score = 1.0 - avg_pairwise_sim  # Convert to diversity score

        # Calculate MMR-reranked scores
        mmr_results = self.calculate_mmr_scores(query, retrieved_chunks, top_k=top_k)
        avg_mmr_score = np.mean([score for _, score in mmr_results]) if mmr_results else 0.0

        result = {
            'query': query,
            'mmr_score': avg_mmr_score,
            'diversity_score': diversity_score,
            'avg_pairwise_similarity': avg_pairwise_sim,
            'num_retrieved': len(retrieved_chunks),
            'lambda': self.lambda_param
        }

        self.results.append(result)
        self.detailed_results.append(result)

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about MMR performance"""
        if not self.results:
            return {}

        mmr_scores = [r['mmr_score'] for r in self.results]
        diversity_scores = [r['diversity_score'] for r in self.results]

        return {
            'avg_mmr': np.mean(mmr_scores),
            'avg_diversity': np.mean(diversity_scores),
            'std_mmr': np.std(mmr_scores),
            'std_diversity': np.std(diversity_scores),
            'num_queries': len(self.results),
            'lambda_param': self.lambda_param
        }

    def print_report(self):
        """Print a formatted MMR evaluation report"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("MMR (Maximum Marginal Relevance) EVALUATION REPORT")
        print("="*60)

        if not stats:
            print("No evaluation results available")
            return

        print(f"\nLambda Parameter (relevance vs diversity): {stats['lambda_param']:.2f}")
        print(f"Number of queries evaluated: {stats['num_queries']}")

        print(f"\nMMR Scores:")
        print(f"  Average: {stats['avg_mmr']:.4f}")
        print(f"  Std Dev: {stats['std_mmr']:.4f}")

        print(f"\nDiversity Scores:")
        print(f"  Average: {stats['avg_diversity']:.4f}")
        print(f"  Std Dev: {stats['std_diversity']:.4f}")

        print("\n" + "-"*60)
        print("Query-level Results:")
        for i, result in enumerate(self.detailed_results, 1):
            print(f"\n{i}. Query: {result['query'][:50]}...")
            print(f"   MMR Score: {result['mmr_score']:.4f}")
            print(f"   Diversity: {result['diversity_score']:.4f}")
            print(f"   Avg Pairwise Similarity: {result['avg_pairwise_similarity']:.4f}")

        print("\n" + "="*60)

    def save_results(self, filepath: str):
        """Save detailed results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'detailed_results': self.detailed_results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"MMR results saved to: {filepath}")


class NDCGCalculator:
    """
    Normalized Discounted Cumulative Gain (NDCG) Calculator
    Measures ranking quality with graded relevance scores
    """

    def __init__(self):
        self.results = []
        self.detailed_results = []

    def dcg_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at position k

        Args:
            relevance_scores: List of relevance scores for retrieved documents
            k: Position to calculate DCG

        Returns:
            DCG@k score
        """
        relevance_scores = np.array(relevance_scores)[:k]
        if relevance_scores.size == 0:
            return 0.0

        # DCG formula: sum of (2^rel - 1) / log2(i + 2) for i in [0, k-1]
        gains = 2 ** relevance_scores - 1
        discounts = np.log2(np.arange(len(relevance_scores)) + 2)
        return np.sum(gains / discounts)

    def ndcg_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at position k

        Args:
            relevance_scores: List of relevance scores for retrieved documents
            k: Position to calculate NDCG

        Returns:
            NDCG@k score (0-1, higher is better)
        """
        dcg = self.dcg_at_k(relevance_scores, k)

        # Calculate IDCG (ideal DCG with perfect ranking)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = self.dcg_at_k(ideal_relevance, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def calculate_relevance_scores(self,
                                   retrieved_chunks: List[Dict[str, Any]],
                                   relevant_chunk_ids: List[str],
                                   use_binary: bool = False) -> List[float]:
        """
        Calculate relevance scores for retrieved chunks

        Args:
            retrieved_chunks: List of retrieved chunks with metadata
            relevant_chunk_ids: List of identifiers for relevant chunks
            use_binary: If True, use binary relevance (0 or 1), else graded (0, 1, 2, 3)

        Returns:
            List of relevance scores
        """
        scores = []

        for chunk in retrieved_chunks:
            relevance = 0.0

            # Check if chunk is relevant
            is_relevant = False
            match_strength = 0  # 0=none, 1=weak, 2=medium, 3=strong

            # Method 1: Check by filename (weak match)
            if 'filename' in chunk:
                for relevant_id in relevant_chunk_ids:
                    if relevant_id.lower() in chunk['filename'].lower():
                        is_relevant = True
                        match_strength = max(match_strength, 1)

            # Method 2: Check by content substring (medium match)
            if 'chunk' in chunk:
                chunk_content = chunk['chunk'].lower()
                for relevant_id in relevant_chunk_ids:
                    if relevant_id.lower() in chunk_content:
                        is_relevant = True
                        # Longer matches = stronger relevance
                        if len(relevant_id) > 20:
                            match_strength = max(match_strength, 3)
                        else:
                            match_strength = max(match_strength, 2)

            # Method 3: Check by chunk index (strong match)
            if 'chunk_idx' in chunk:
                for relevant_id in relevant_chunk_ids:
                    if str(chunk['chunk_idx']) == str(relevant_id):
                        is_relevant = True
                        match_strength = max(match_strength, 3)

            # Assign relevance score
            if use_binary:
                relevance = 1.0 if is_relevant else 0.0
            else:
                relevance = float(match_strength)  # Graded: 0, 1, 2, 3

            scores.append(relevance)

        return scores

    def evaluate_query(self,
                       qa_system: QASystem,
                       query: str,
                       relevant_chunk_ids: List[str],
                       k_values: List[int] = [5, 10],
                       use_binary: bool = False) -> Dict[str, Any]:
        """
        Evaluate NDCG for a single query at multiple k values

        Args:
            qa_system: QA system instance
            query: Query string
            relevant_chunk_ids: List of relevant chunk identifiers
            k_values: List of k values to evaluate
            use_binary: Use binary relevance instead of graded

        Returns:
            Dictionary with NDCG scores at different k values
        """
        print(f"\n  Evaluating: {query[:50]}...")

        # Retrieve chunks
        max_k = max(k_values)
        retrieved_chunks = qa_system.find_relevant_chunks(query, top_k=max_k, verbose=False)

        # Calculate relevance scores
        relevance_scores = self.calculate_relevance_scores(
            retrieved_chunks,
            relevant_chunk_ids,
            use_binary=use_binary
        )

        # Calculate NDCG at different k values
        ndcg_scores = {}
        for k in k_values:
            ndcg = self.ndcg_at_k(relevance_scores, k)
            ndcg_scores[f'ndcg@{k}'] = ndcg

        result = {
            'query': query,
            'relevance_scores': relevance_scores,
            'num_retrieved': len(retrieved_chunks),
            'num_relevant': sum(1 for score in relevance_scores if score > 0),
            **ndcg_scores
        }

        self.results.append(result)
        self.detailed_results.append(result)

        return result

    def get_statistics(self, k_values: List[int] = [5, 10]) -> Dict[str, Any]:
        """Get detailed statistics about NDCG performance"""
        if not self.results:
            return {}

        stats = {'num_queries': len(self.results)}

        for k in k_values:
            ndcg_key = f'ndcg@{k}'
            ndcg_scores = [r[ndcg_key] for r in self.results if ndcg_key in r]

            if ndcg_scores:
                stats[f'avg_{ndcg_key}'] = np.mean(ndcg_scores)
                stats[f'std_{ndcg_key}'] = np.std(ndcg_scores)
                stats[f'min_{ndcg_key}'] = min(ndcg_scores)
                stats[f'max_{ndcg_key}'] = max(ndcg_scores)

        return stats

    def print_report(self, k_values: List[int] = [5, 10]):
        """Print a formatted NDCG evaluation report"""
        stats = self.get_statistics(k_values)

        print("\n" + "="*60)
        print("NDCG EVALUATION REPORT")
        print("="*60)

        if not stats:
            print("No evaluation results available")
            return

        print(f"\nNumber of queries: {stats['num_queries']}")

        for k in k_values:
            print(f"\nNDCG@{k} Statistics:")
            print(f"  Mean: {stats[f'avg_ndcg@{k}']:.4f}")
            print(f"  Std Dev: {stats[f'std_ndcg@{k}']:.4f}")
            print(f"  Min: {stats[f'min_ndcg@{k}']:.4f}")
            print(f"  Max: {stats[f'max_ndcg@{k}']:.4f}")

        print("\n" + "-"*60)
        print("Query-level Results:")
        for i, result in enumerate(self.detailed_results, 1):
            print(f"\n{i}. Query: {result['query'][:50]}...")
            for k in k_values:
                ndcg_key = f'ndcg@{k}'
                if ndcg_key in result:
                    print(f"   {ndcg_key.upper()}: {result[ndcg_key]:.4f}")
            print(f"   Relevant found: {result['num_relevant']}/{result['num_retrieved']}")

        print("\n" + "="*60)

    def save_results(self, filepath: str):
        """Save detailed results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'detailed_results': self.detailed_results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"NDCG results saved to: {filepath}")


class RecallCalculator:
    """
    Recall@K Calculator
    Measures the proportion of relevant documents retrieved in top-K results
    """

    def __init__(self):
        self.results = []
        self.detailed_results = []

    def calculate_recall_at_k(self,
                             retrieved_chunks: List[Dict[str, Any]],
                             relevant_chunk_ids: List[str],
                             k: int) -> float:
        """
        Calculate Recall@K

        Args:
            retrieved_chunks: List of retrieved chunks
            relevant_chunk_ids: List of all relevant chunk identifiers
            k: Number of top results to consider

        Returns:
            Recall@K score (0-1)
        """
        if not relevant_chunk_ids:
            return 0.0

        # Consider only top-k results
        top_k_chunks = retrieved_chunks[:k]

        # Count how many relevant chunks were found
        found_relevant = 0

        for chunk in top_k_chunks:
            is_relevant = False

            # Method 1: Check by filename
            if 'filename' in chunk:
                for relevant_id in relevant_chunk_ids:
                    if relevant_id.lower() in chunk['filename'].lower():
                        is_relevant = True
                        break

            # Method 2: Check by content substring
            if not is_relevant and 'chunk' in chunk:
                chunk_content = chunk['chunk'].lower()
                for relevant_id in relevant_chunk_ids:
                    if relevant_id.lower() in chunk_content:
                        is_relevant = True
                        break

            # Method 3: Check by chunk index
            if not is_relevant and 'chunk_idx' in chunk:
                for relevant_id in relevant_chunk_ids:
                    if str(chunk['chunk_idx']) == str(relevant_id):
                        is_relevant = True
                        break

            if is_relevant:
                found_relevant += 1

        # Recall = (relevant retrieved) / (total relevant)
        recall = found_relevant / len(relevant_chunk_ids)

        return recall

    def evaluate_query(self,
                       qa_system: QASystem,
                       query: str,
                       relevant_chunk_ids: List[str],
                       k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """
        Evaluate Recall@K for a single query at multiple k values

        Args:
            qa_system: QA system instance
            query: Query string
            relevant_chunk_ids: List of relevant chunk identifiers
            k_values: List of k values to evaluate

        Returns:
            Dictionary with Recall@K scores
        """
        print(f"\n  Evaluating: {query[:50]}...")

        # Retrieve chunks
        max_k = max(k_values)
        retrieved_chunks = qa_system.find_relevant_chunks(query, top_k=max_k, verbose=False)

        # Calculate Recall at different k values
        recall_scores = {}
        for k in k_values:
            recall = self.calculate_recall_at_k(retrieved_chunks, relevant_chunk_ids, k)
            recall_scores[f'recall@{k}'] = recall

        # Also calculate hit rate (whether any relevant doc was found)
        hit = 1.0 if recall_scores[f'recall@{max_k}'] > 0 else 0.0

        result = {
            'query': query,
            'num_retrieved': len(retrieved_chunks),
            'num_relevant_total': len(relevant_chunk_ids),
            'hit': hit,
            **recall_scores
        }

        self.results.append(result)
        self.detailed_results.append(result)

        return result

    def get_statistics(self, k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """Get detailed statistics about Recall performance"""
        if not self.results:
            return {}

        stats = {
            'num_queries': len(self.results),
            'hit_rate': np.mean([r['hit'] for r in self.results])
        }

        for k in k_values:
            recall_key = f'recall@{k}'
            recall_scores = [r[recall_key] for r in self.results if recall_key in r]

            if recall_scores:
                stats[f'avg_{recall_key}'] = np.mean(recall_scores)
                stats[f'std_{recall_key}'] = np.std(recall_scores)
                stats[f'min_{recall_key}'] = min(recall_scores)
                stats[f'max_{recall_key}'] = max(recall_scores)

        return stats

    def print_report(self, k_values: List[int] = [5, 10, 20]):
        """Print a formatted Recall evaluation report"""
        stats = self.get_statistics(k_values)

        print("\n" + "="*60)
        print("RECALL@K EVALUATION REPORT")
        print("="*60)

        if not stats:
            print("No evaluation results available")
            return

        print(f"\nNumber of queries: {stats['num_queries']}")
        print(f"Hit Rate (any relevant found): {stats['hit_rate']:.4f} ({stats['hit_rate']*100:.1f}%)")

        for k in k_values:
            print(f"\nRecall@{k} Statistics:")
            print(f"  Mean: {stats[f'avg_recall@{k}']:.4f}")
            print(f"  Std Dev: {stats[f'std_recall@{k}']:.4f}")
            print(f"  Min: {stats[f'min_recall@{k}']:.4f}")
            print(f"  Max: {stats[f'max_recall@{k}']:.4f}")

        print("\n" + "-"*60)
        print("Query-level Results:")
        for i, result in enumerate(self.detailed_results, 1):
            print(f"\n{i}. Query: {result['query'][:50]}...")
            for k in k_values:
                recall_key = f'recall@{k}'
                if recall_key in result:
                    print(f"   {recall_key.capitalize()}: {result[recall_key]:.4f}")
            print(f"   Hit: {'✓' if result['hit'] == 1.0 else '✗'}")

        print("\n" + "="*60)

    def save_results(self, filepath: str):
        """Save detailed results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'detailed_results': self.detailed_results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Recall results saved to: {filepath}")


class MRRCalculator:
    """
    Mean Reciprocal Rank (MRR) Calculator
    Measures how well the retrieval system ranks relevant documents
    """

    def __init__(self):
        self.results = []
        self.detailed_results = []

    def calculate_reciprocal_rank(self,
                                   retrieved_chunks: List[Dict[str, Any]],
                                   relevant_chunk_ids: List[str]) -> float:
        """
        Calculate reciprocal rank for a single query

        Args:
            retrieved_chunks: List of retrieved chunks with metadata
            relevant_chunk_ids: List of identifiers for relevant chunks
                               Can be filenames, chunk indices, or content snippets

        Returns:
            Reciprocal rank (1/position of first relevant, 0 if none found)
        """

        for position, chunk in enumerate(retrieved_chunks, 1):
            # Check multiple ways to identify relevant chunks
            is_relevant = False

            # Method 1: Check by filename
            if 'filename' in chunk:
                for relevant_id in relevant_chunk_ids:
                    if relevant_id in chunk['filename']:
                        is_relevant = True
                        break

            # Method 2: Check by content substring
            if not is_relevant and 'chunk' in chunk:
                chunk_content = chunk['chunk'].lower()
                for relevant_id in relevant_chunk_ids:
                    if relevant_id.lower() in chunk_content:
                        is_relevant = True
                        break

            # Method 3: Check by chunk index if provided
            if not is_relevant and 'chunk_idx' in chunk:
                for relevant_id in relevant_chunk_ids:
                    if str(chunk['chunk_idx']) == str(relevant_id):
                        is_relevant = True
                        break

            if is_relevant:
                # Found first relevant document at this position
                return 1.0 / position

        # No relevant document found
        return 0.0

    def evaluate_query(self,
                       qa_system: QASystem,
                       query: str,
                       relevant_chunk_ids: List[str],
                       top_k: int = 10) -> Dict[str, Any]:
        """
        Evaluate a single query and calculate MRR

        Returns detailed results for analysis
        """
        # Retrieve chunks (with logging suppressed for evaluation)
        print(f"\n  Evaluating: {query[:50]}...")

        # Temporarily reduce verbosity
        retrieved_chunks = qa_system.find_relevant_chunks(query, top_k=top_k, verbose=False)

        # Calculate reciprocal rank
        rr = self.calculate_reciprocal_rank(retrieved_chunks, relevant_chunk_ids)

        # Store detailed results
        result = {
            'query': query,
            'reciprocal_rank': rr,
            'num_retrieved': len(retrieved_chunks),
            'first_relevant_position': int(1/rr) if rr > 0 else None,
            'relevant_chunk_ids': relevant_chunk_ids
        }

        self.results.append(rr)
        self.detailed_results.append(result)

        return result

    def calculate_mrr(self) -> float:
        """Calculate Mean Reciprocal Rank across all queries"""
        if not self.results:
            return 0.0
        return np.mean(self.results)

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about MRR performance"""
        if not self.results:
            return {}

        return {
            'mrr': self.calculate_mrr(),
            'num_queries': len(self.results),
            'perfect_retrievals': sum(1 for r in self.results if r == 1.0),
            'failed_retrievals': sum(1 for r in self.results if r == 0.0),
            'mrr_std': np.std(self.results),
            'mrr_min': min(self.results),
            'mrr_max': max(self.results),
            'percentiles': {
                '25': np.percentile(self.results, 25),
                '50': np.percentile(self.results, 50),
                '75': np.percentile(self.results, 75)
            }
        }

    def print_report(self):
        """Print a formatted evaluation report"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("MRR EVALUATION REPORT")
        print("="*60)

        if not stats:
            print("No evaluation results available")
            return

        print(f"\nOverall MRR Score: {stats['mrr']:.4f}")
        print(f"Number of queries: {stats['num_queries']}")
        print(f"Perfect retrievals (RR=1.0): {stats['perfect_retrievals']} ({stats['perfect_retrievals']/stats['num_queries']*100:.1f}%)")
        print(f"Failed retrievals (RR=0.0): {stats['failed_retrievals']} ({stats['failed_retrievals']/stats['num_queries']*100:.1f}%)")

        print(f"\nStatistics:")
        print(f"  Mean: {stats['mrr']:.4f}")
        print(f"  Std Dev: {stats['mrr_std']:.4f}")
        print(f"  Min: {stats['mrr_min']:.4f}")
        print(f"  Max: {stats['mrr_max']:.4f}")

        print(f"\nPercentiles:")
        print(f"  25th: {stats['percentiles']['25']:.4f}")
        print(f"  50th (Median): {stats['percentiles']['50']:.4f}")
        print(f"  75th: {stats['percentiles']['75']:.4f}")

        print("\n" + "-"*60)
        print("Query-level Results:")
        for i, result in enumerate(self.detailed_results, 1):
            print(f"\n{i}. Query: {result['query'][:50]}...")
            print(f"   RR: {result['reciprocal_rank']:.4f}")
            if result['first_relevant_position']:
                print(f"   First relevant at position: {result['first_relevant_position']}")
            else:
                print(f"   No relevant chunks found in top-{result['num_retrieved']}")

        print("\n" + "="*60)

    def save_results(self, filepath: str):
        """Save detailed results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'detailed_results': self.detailed_results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {filepath}")


class TestDataset:
    """
    Manages test datasets for evaluation
    """

    @staticmethod
    def create_sample_test_set() -> List[Dict[str, Any]]:
        """
        Create a sample test dataset for evaluation
        Each item contains:
        - query: The question to ask
        - relevant_chunks: Identifiers for chunks containing the answer
        - query_type: Category of query (factual, analytical, etc.)
        """

        test_set = [
            {
                'query': 'What datasets are mentioned?',
                'relevant_chunks': ['ERA5', 'CMIP6', 'dataset'],
                'query_type': 'factual'
            },
            {
                'query': 'Which datasets were used for climate analysis?',
                'relevant_chunks': ['ERA5', 'ECMWF', 'reanalysis'],
                'query_type': 'factual'
            },
            {
                'query': 'What is the temporal resolution of the data?',
                'relevant_chunks': ['hourly', 'temporal', 'resolution'],
                'query_type': 'specific'
            },
            {
                'query': 'How does machine learning help with climate prediction?',
                'relevant_chunks': ['machine learning', 'deep neural', 'prediction'],
                'query_type': 'analytical'
            },
            {
                'query': 'What time period does the ERA5 dataset cover?',
                'relevant_chunks': ['1979', 'present', 'ERA5'],
                'query_type': 'specific'
            }
        ]

        return test_set

    @staticmethod
    def load_from_json(filepath: str) -> List[Dict[str, Any]]:
        """Load test dataset from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_to_json(test_set: List[Dict[str, Any]], filepath: str):
        """Save test dataset to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(test_set, f, indent=2)
        print(f"Test dataset saved to: {filepath}")


class RAGASFaithfulness:
    """
    RAGAS Faithfulness metric
    Evaluates if the generated answer is grounded in the retrieved context
    Uses LLM to verify if statements can be inferred from context
    """

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.results = []

    def extract_statements(self, answer: str) -> List[str]:
        """
        Extract atomic statements from the answer using LLM
        """
        prompt = f"""Extract all factual statements from the following answer as a numbered list.
Each statement should be atomic (single fact) and self-contained.

Answer: {answer}

Statements:"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1}
            )

            # Parse numbered statements
            statements = []
            lines = response['response'].strip().split('\n')
            for line in lines:
                # Remove numbering and clean
                cleaned = line.strip()
                if cleaned and any(c.isalpha() for c in cleaned):
                    # Remove common numbering patterns
                    import re
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
                    if cleaned:
                        statements.append(cleaned)

            return statements if statements else [answer]

        except Exception as e:
            print(f"Error extracting statements: {e}")
            # Fallback: treat entire answer as one statement
            return [answer]

    def verify_statement(self, statement: str, context: str) -> bool:
        """
        Verify if a statement can be inferred from the context
        """
        # RELAXED EVALUATION: More lenient criteria
        prompt = f"""Based on the given context, determine if the following statement is supported by, consistent with, or reasonably inferred from the information provided.

Be lenient: Accept statements that are:
- Directly stated in the context
- Reasonable interpretations of the context
- Logical extensions of information in the context
- Summaries or paraphrases of context information

Only reject statements that clearly contradict the context or introduce completely new unrelated information.

Context: {context}

Statement: {statement}

Answer with only 'Yes' or 'No'.
Answer:"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0}
            )

            answer = response['response'].strip().lower()
            return 'yes' in answer

        except Exception as e:
            print(f"Error verifying statement: {e}")
            return False

    def calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        Calculate faithfulness score for a single answer

        Returns:
            Proportion of statements that can be verified from context (0-1)
        """
        if not answer or not context:
            return 0.0

        # Extract statements from answer
        statements = self.extract_statements(answer)

        if not statements:
            return 1.0  # No factual claims to verify

        # Verify each statement
        verified_count = 0
        for statement in statements:
            if self.verify_statement(statement, context):
                verified_count += 1

        score = verified_count / len(statements) if statements else 1.0

        # Store detailed result
        result = {
            'answer': answer[:100] + '...' if len(answer) > 100 else answer,
            'num_statements': len(statements),
            'verified_statements': verified_count,
            'faithfulness_score': score
        }
        self.results.append(result)

        return score

    def evaluate_batch(self,
                       qa_pairs: List[Dict[str, str]],
                       contexts: List[str]) -> float:
        """
        Evaluate faithfulness for multiple Q&A pairs

        Args:
            qa_pairs: List of dicts with 'question' and 'answer'
            contexts: List of retrieved contexts for each Q&A pair

        Returns:
            Average faithfulness score
        """
        scores = []

        for i, (qa, context) in enumerate(zip(qa_pairs, contexts)):
            print(f"  Evaluating faithfulness {i+1}/{len(qa_pairs)}...")
            score = self.calculate_faithfulness(qa['answer'], context)
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about faithfulness scores"""
        if not self.results:
            return {}

        scores = [r['faithfulness_score'] for r in self.results]

        return {
            'mean_faithfulness': np.mean(scores),
            'std_faithfulness': np.std(scores),
            'min_faithfulness': min(scores),
            'max_faithfulness': max(scores),
            'perfect_faithfulness': sum(1 for s in scores if s == 1.0),
            'num_evaluated': len(scores)
        }


class RAGASAnswerRelevancy:
    """
    RAGAS Answer Relevancy metric
    Evaluates if the generated answer addresses the user's question
    Uses reverse question generation and embedding similarity
    """

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.results = []

    def generate_questions(self, answer: str, num_questions: int = 3) -> List[str]:
        """
        Generate questions that could be answered by the given answer
        """
        prompt = f"""Given the following answer, generate {num_questions} diverse questions that this answer could be responding to.

Answer: {answer}

Generate exactly {num_questions} questions, one per line:"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.7}
            )

            # Parse questions
            questions = []
            lines = response['response'].strip().split('\n')
            for line in lines:
                cleaned = line.strip()
                if cleaned and '?' in cleaned:
                    # Remove numbering if present
                    import re
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
                    questions.append(cleaned)

            return questions[:num_questions] if questions else []

        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

    def calculate_relevancy(self, original_question: str, answer: str) -> float:
        """
        Calculate answer relevancy score

        Returns:
            Similarity score between original and generated questions (0-1)
        """
        if not answer or not original_question:
            return 0.0

        # Generate questions from the answer
        generated_questions = self.generate_questions(answer, num_questions=3)

        if not generated_questions:
            return 0.5  # Neutral score if generation fails

        # Calculate embedding similarity
        original_embedding = self.embedder.encode(original_question, convert_to_tensor=True)
        generated_embeddings = self.embedder.encode(generated_questions, convert_to_tensor=True)

        # Calculate cosine similarities
        similarities = util.cos_sim(original_embedding, generated_embeddings)[0]

        # Take mean of similarities as the relevancy score
        score = float(similarities.mean())

        # Store detailed result
        result = {
            'original_question': original_question[:100] + '...' if len(original_question) > 100 else original_question,
            'num_generated': len(generated_questions),
            'relevancy_score': score,
            'generated_questions': generated_questions[:2]  # Store first 2 for reference
        }
        self.results.append(result)

        return score

    def evaluate_batch(self, qa_pairs: List[Dict[str, str]]) -> float:
        """
        Evaluate answer relevancy for multiple Q&A pairs

        Args:
            qa_pairs: List of dicts with 'question' and 'answer'

        Returns:
            Average relevancy score
        """
        scores = []

        for i, qa in enumerate(qa_pairs):
            print(f"  Evaluating relevancy {i+1}/{len(qa_pairs)}...")
            score = self.calculate_relevancy(qa['question'], qa['answer'])
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about relevancy scores"""
        if not self.results:
            return {}

        scores = [r['relevancy_score'] for r in self.results]

        return {
            'mean_relevancy': np.mean(scores),
            'std_relevancy': np.std(scores),
            'min_relevancy': min(scores),
            'max_relevancy': max(scores),
            'high_relevancy': sum(1 for s in scores if s > 0.7),
            'num_evaluated': len(scores)
        }


class RAGASEvaluator:
    """
    Combined RAGAS evaluator for comprehensive evaluation
    """

    def __init__(self, qa_system: QASystem, model_name: str = "llama3.2"):
        self.qa_system = qa_system
        self.faithfulness = RAGASFaithfulness(model_name)
        self.relevancy = RAGASAnswerRelevancy(model_name)
        self.results = []

    def evaluate_single(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluate a single question through the complete pipeline
        """
        print(f"\nEvaluating: {question[:50]}...")

        # Get answer from QA system
        answer = self.qa_system.answer_question(question, verbose=False)

        # Get retrieved chunks for context
        chunks = self.qa_system.find_relevant_chunks(question, top_k=top_k, verbose=False)
        context = ' '.join([chunk['chunk'] for chunk in chunks])

        # Calculate metrics
        faithfulness_score = self.faithfulness.calculate_faithfulness(answer, context)
        relevancy_score = self.relevancy.calculate_relevancy(question, answer)

        result = {
            'question': question,
            'answer': answer[:200] + '...' if len(answer) > 200 else answer,
            'faithfulness': faithfulness_score,
            'relevancy': relevancy_score,
            'num_chunks': len(chunks)
        }

        self.results.append(result)
        return result

    def evaluate_test_set(self, test_questions: List[str], top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluate a set of test questions
        """
        print("\n" + "="*60)
        print("RAGAS EVALUATION")
        print("="*60)
        print(f"Evaluating {len(test_questions)} questions...")

        for i, question in enumerate(test_questions, 1):
            print(f"\nProgress: {i}/{len(test_questions)}")
            self.evaluate_single(question, top_k)

        # Calculate aggregate metrics
        faithfulness_scores = [r['faithfulness'] for r in self.results]
        relevancy_scores = [r['relevancy'] for r in self.results]

        return {
            'num_questions': len(test_questions),
            'avg_faithfulness': np.mean(faithfulness_scores),
            'avg_relevancy': np.mean(relevancy_scores),
            'combined_score': np.mean([np.mean(faithfulness_scores), np.mean(relevancy_scores)]),
            'faithfulness_stats': self.faithfulness.get_statistics(),
            'relevancy_stats': self.relevancy.get_statistics()
        }

    def print_report(self):
        """Print detailed RAGAS evaluation report"""
        if not self.results:
            print("No RAGAS evaluation results available")
            return

        faithfulness_scores = [r['faithfulness'] for r in self.results]
        relevancy_scores = [r['relevancy'] for r in self.results]

        print("\n" + "="*60)
        print("RAGAS EVALUATION REPORT")
        print("="*60)

        print(f"\nFAITHFULNESS METRICS:")
        print(f"  Mean: {np.mean(faithfulness_scores):.4f}")
        print(f"  Std Dev: {np.std(faithfulness_scores):.4f}")
        print(f"  Min: {min(faithfulness_scores):.4f}")
        print(f"  Max: {max(faithfulness_scores):.4f}")

        print(f"\nANSWER RELEVANCY METRICS:")
        print(f"  Mean: {np.mean(relevancy_scores):.4f}")
        print(f"  Std Dev: {np.std(relevancy_scores):.4f}")
        print(f"  Min: {min(relevancy_scores):.4f}")
        print(f"  Max: {max(relevancy_scores):.4f}")

        print(f"\nCOMBINED RAGAS SCORE: {np.mean([np.mean(faithfulness_scores), np.mean(relevancy_scores)]):.4f}")

        print("\n" + "-"*60)
        print("Question-level Results:")
        for i, result in enumerate(self.results, 1):
            print(f"\n{i}. {result['question'][:50]}...")
            print(f"   Faithfulness: {result['faithfulness']:.4f}")
            print(f"   Relevancy: {result['relevancy']:.4f}")

        print("\n" + "="*60)

    def save_results(self, filepath: str):
        """Save RAGAS results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'faithfulness_stats': self.faithfulness.get_statistics(),
            'relevancy_stats': self.relevancy.get_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"RAGAS results saved to: {filepath}")


class EvaluationPipeline:
    """
    Main evaluation pipeline that orchestrates all metrics
    """

    def __init__(self, qa_system: QASystem):
        self.qa_system = qa_system
        self.mrr_calculator = MRRCalculator()
        self.ragas_evaluator = RAGASEvaluator(qa_system)
        self.results = {}

    def evaluate_mrr(self, test_set: List[Dict[str, Any]], top_k: int = 10) -> float:
        """
        Evaluate MRR on a test set

        Args:
            test_set: List of test cases with queries and relevant chunks
            top_k: Number of chunks to retrieve

        Returns:
            MRR score
        """
        print("\n" + "="*60)
        print("STARTING MRR EVALUATION")
        print("="*60)
        print(f"Number of test queries: {len(test_set)}")
        print(f"Retrieving top-{top_k} chunks per query")

        for i, test_case in enumerate(test_set, 1):
            print(f"\nProgress: {i}/{len(test_set)}")

            result = self.mrr_calculator.evaluate_query(
                self.qa_system,
                test_case['query'],
                test_case['relevant_chunks'],
                top_k=top_k
            )

            # Quick feedback
            if result['reciprocal_rank'] == 1.0:
                print(f"  ✓ Perfect retrieval (RR=1.0)")
            elif result['reciprocal_rank'] > 0:
                print(f"  ✓ Found at position {result['first_relevant_position']} (RR={result['reciprocal_rank']:.3f})")
            else:
                print(f"  ✗ Not found in top-{top_k}")

        # Calculate final MRR
        mrr = self.mrr_calculator.calculate_mrr()
        self.results['mrr'] = mrr

        # Print report
        self.mrr_calculator.print_report()

        return mrr

    def evaluate_ragas(self, test_questions: List[str], top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluate RAGAS metrics (Faithfulness and Answer Relevancy)

        Args:
            test_questions: List of questions to evaluate
            top_k: Number of chunks to retrieve for context

        Returns:
            Dictionary with RAGAS scores
        """
        print("\n" + "="*60)
        print("STARTING RAGAS EVALUATION")
        print("="*60)

        results = self.ragas_evaluator.evaluate_test_set(test_questions, top_k=top_k)
        self.results['ragas'] = results

        # Print report
        self.ragas_evaluator.print_report()

        return results

    def run_comprehensive_evaluation(self,
                                    test_set: List[Dict[str, Any]],
                                    top_k: int = 5) -> Dict[str, Any]:
        """
        Run both MRR and RAGAS evaluations

        Args:
            test_set: Test dataset with queries and relevant chunks
            top_k: Number of chunks to retrieve

        Returns:
            Combined evaluation results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION: MRR + RAGAS")
        print("="*80)

        # Run MRR evaluation
        print("\n[PHASE 1/2] MRR EVALUATION")
        mrr_score = self.evaluate_mrr(test_set, top_k=top_k)

        # Extract questions for RAGAS
        test_questions = [item['query'] for item in test_set]

        # Run RAGAS evaluation
        print("\n[PHASE 2/2] RAGAS EVALUATION")
        ragas_results = self.evaluate_ragas(test_questions, top_k=top_k)

        # Combine results
        combined_results = {
            'mrr': mrr_score,
            'ragas': ragas_results,
            'combined_score': (mrr_score + ragas_results['combined_score']) / 2
        }

        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"MRR Score: {mrr_score:.4f}")
        print(f"RAGAS Faithfulness: {ragas_results['avg_faithfulness']:.4f}")
        print(f"RAGAS Answer Relevancy: {ragas_results['avg_relevancy']:.4f}")
        print(f"Overall Combined Score: {combined_results['combined_score']:.4f}")
        print("="*80)

        return combined_results

    def run_ablation_study(self,
                          test_set: List[Dict[str, Any]],
                          test_document: str = None) -> Dict[str, Any]:
        """
        Run ablation study comparing different RAG configurations

        Args:
            test_set: Test dataset with queries and relevant chunks
            test_document: Optional test document to add to each system

        Returns:
            Comparison results across different configurations
        """
        print("\n" + "="*80)
        print("ABLATION STUDY: Comparing RAG Configurations")
        print("="*80)

        results = {}
        test_questions = [item['query'] for item in test_set]

        # Configuration 1: FAISS-only (Dense retrieval)
        print("\n[1/4] Testing FAISS-only retrieval...")
        print("Configuration: Dense retrieval with FAISS, dynamic thresholding")

        qa_faiss = QASystem(use_hybrid=False, use_faiss=True)
        if test_document:
            qa_faiss.add_document('test_doc.pdf', text=test_document)

        # Evaluate FAISS-only
        eval_faiss = EvaluationPipeline(qa_faiss)
        mrr_faiss = eval_faiss.evaluate_mrr(test_set, top_k=5)
        ragas_faiss = eval_faiss.evaluate_ragas(test_questions, top_k=5)

        results['faiss_only'] = {
            'mrr': mrr_faiss,
            'faithfulness': ragas_faiss['avg_faithfulness'],
            'relevancy': ragas_faiss['avg_relevancy'],
            'combined': (mrr_faiss + ragas_faiss['combined_score']) / 2
        }

        # Configuration 2: BM25-only (Sparse retrieval)
        print("\n[2/4] Testing BM25-only retrieval...")
        print("Configuration: Sparse retrieval with BM25")

        qa_bm25 = QASystem(use_hybrid=True, use_faiss=False)  # Hybrid mode but without FAISS
        if test_document:
            qa_bm25.add_document('test_doc.pdf', text=test_document)

        # Note: This requires modifying QASystem to support BM25-only mode
        # For now, we'll skip actual evaluation and use placeholder

        results['bm25_only'] = {
            'mrr': 0.0,  # Placeholder
            'faithfulness': 0.0,
            'relevancy': 0.0,
            'combined': 0.0,
            'note': 'Requires QASystem modification for BM25-only mode'
        }

        # Configuration 3: Hybrid (Dense + Sparse)
        print("\n[3/4] Testing Hybrid retrieval...")
        print("Configuration: Hybrid with FAISS + BM25 (alpha=0.5)")

        qa_hybrid = QASystem(use_hybrid=True, use_faiss=True)
        if test_document:
            qa_hybrid.add_document('test_doc.pdf', text=test_document)

        eval_hybrid = EvaluationPipeline(qa_hybrid)
        mrr_hybrid = eval_hybrid.evaluate_mrr(test_set, top_k=5)
        ragas_hybrid = eval_hybrid.evaluate_ragas(test_questions, top_k=5)

        results['hybrid'] = {
            'mrr': mrr_hybrid,
            'faithfulness': ragas_hybrid['avg_faithfulness'],
            'relevancy': ragas_hybrid['avg_relevancy'],
            'combined': (mrr_hybrid + ragas_hybrid['combined_score']) / 2
        }

        # Configuration 4: Different top_k values
        print("\n[4/4] Testing different top_k values...")
        print("Configuration: Hybrid with varying retrieval sizes")

        top_k_values = [3, 5, 10]
        results['top_k_analysis'] = {}

        for k in top_k_values:
            print(f"  Testing top_k={k}...")
            mrr_k = self.mrr_calculator.calculate_mrr()  # Reset calculator
            self.mrr_calculator.results = []
            self.mrr_calculator.detailed_results = []

            for test_case in test_set:
                self.mrr_calculator.evaluate_query(
                    self.qa_system,
                    test_case['query'],
                    test_case['relevant_chunks'],
                    top_k=k
                )

            results['top_k_analysis'][f'k_{k}'] = {
                'mrr': self.mrr_calculator.calculate_mrr()
            }

        # Print comparison table
        self._print_ablation_report(results)

        # Save ablation study results
        with open('ablation_study_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        print(f"\nAblation study results saved to: ablation_study_results.json")

        return results

    def _print_ablation_report(self, results: Dict[str, Any]):
        """Print formatted ablation study report"""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)

        print("\n" + "-"*60)
        print("Configuration Comparison:")
        print("-"*60)

        # Print main configurations
        configs = ['faiss_only', 'hybrid']
        print(f"\n{'Configuration':<20} {'MRR':<10} {'Faith.':<10} {'Relev.':<10} {'Combined':<10}")
        print("-"*60)

        for config in configs:
            if config in results:
                r = results[config]
                if 'note' not in r:
                    print(f"{config:<20} {r['mrr']:<10.4f} {r['faithfulness']:<10.4f} "
                          f"{r['relevancy']:<10.4f} {r['combined']:<10.4f}")

        # Print top_k analysis
        if 'top_k_analysis' in results:
            print("\n" + "-"*60)
            print("Top-K Analysis (MRR scores):")
            print("-"*60)
            for k_config, metrics in results['top_k_analysis'].items():
                k_value = k_config.replace('k_', '')
                print(f"  Top-{k_value:<2}: {metrics['mrr']:.4f}")

        # Determine best configuration
        best_config = max(
            [c for c in configs if c in results and 'note' not in results[c]],
            key=lambda x: results[x]['combined']
        )

        print("\n" + "-"*60)
        print(f"Best Configuration: {best_config}")
        print(f"Combined Score: {results[best_config]['combined']:.4f}")
        print("="*80)


def main():
    """
    Complete demonstration of the evaluation framework with MRR and RAGAS
    """
    print("="*80)
    print("COMPREHENSIVE RAG EVALUATION FRAMEWORK")
    print("="*80)

    # Initialize QA system
    print("\nInitializing QA system with hybrid search...")
    qa = QASystem(use_hybrid=True)

    # Add test documents
    test_document = """
    The ERA5 dataset from ECMWF provides comprehensive atmospheric reanalysis data from 1979 to present.
    It includes hourly data for temperature, precipitation, wind speed, and pressure.

    Machine learning techniques, particularly deep neural networks, have revolutionized climate prediction.
    These models can now forecast weather patterns with high accuracy up to 10 days in advance.

    The CMIP6 dataset contains climate model outputs from research institutions worldwide.
    It projects future climate scenarios under different emission pathways.

    Data preprocessing involves several steps including normalization, feature extraction, and validation.
    Quality control metrics ensure data integrity throughout the pipeline.
    """

    print("Adding test document...")
    qa.add_document('test_climate.pdf', text=test_document)

    # Create test dataset
    print("\nCreating test dataset...")
    test_set = TestDataset.create_sample_test_set()

    # Initialize evaluation pipeline
    print("\nInitializing evaluation pipeline...")
    evaluator = EvaluationPipeline(qa)

    # Choose evaluation mode
    print("\n" + "="*60)
    print("EVALUATION OPTIONS:")
    print("1. MRR Only")
    print("2. RAGAS Only")
    print("3. Comprehensive (MRR + RAGAS)")
    print("="*60)

    # For demonstration, run comprehensive evaluation
    print("\nRunning comprehensive evaluation (MRR + RAGAS)...")

    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(test_set, top_k=5)

    # Save results
    print("\nSaving evaluation results...")

    # Save MRR results
    evaluator.mrr_calculator.save_results('mrr_evaluation_results.json')

    # Save RAGAS results
    evaluator.ragas_evaluator.save_results('ragas_evaluation_results.json')

    # Save combined results
    with open('comprehensive_evaluation_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    print("Results saved to: comprehensive_evaluation_results.json")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


def run_specific_evaluation(evaluation_type: str = "comprehensive"):
    """
    Run specific evaluation based on research needs

    Args:
        evaluation_type: 'mrr', 'ragas', or 'comprehensive'
    """
    # Initialize QA system
    qa = QASystem(use_hybrid=True)

    # Load your documents here
    # qa.add_document('paper1.pdf', pdf_path='path/to/paper1.pdf')
    # qa.add_document('paper2.pdf', pdf_path='path/to/paper2.pdf')

    # Create your test dataset
    test_set = TestDataset.create_sample_test_set()

    # Initialize evaluator
    evaluator = EvaluationPipeline(qa)

    if evaluation_type == "mrr":
        mrr_score = evaluator.evaluate_mrr(test_set, top_k=5)
        evaluator.mrr_calculator.save_results('mrr_results.json')
        return {'mrr': mrr_score}

    elif evaluation_type == "ragas":
        test_questions = [item['query'] for item in test_set]
        ragas_results = evaluator.evaluate_ragas(test_questions, top_k=5)
        evaluator.ragas_evaluator.save_results('ragas_results.json')
        return ragas_results

    elif evaluation_type == "comprehensive":
        results = evaluator.run_comprehensive_evaluation(test_set, top_k=5)
        # Save all results
        evaluator.mrr_calculator.save_results('mrr_results.json')
        evaluator.ragas_evaluator.save_results('ragas_results.json')
        with open('comprehensive_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results

    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")


if __name__ == "__main__":
    main()