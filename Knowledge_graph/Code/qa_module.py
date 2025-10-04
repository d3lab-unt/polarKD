"""
Q&A Module for PDF Knowledge Explorer
Implements RAG (Retrieval Augmented Generation) using uploaded PDFs
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF - single PDF library
import re
from collections import defaultdict
import pickle
import tiktoken  # For accurate token counting
from datetime import datetime
from rank_bm25 import BM25Okapi  # For hybrid search

# Try to import FAISS - make it optional
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available - using fallback similarity search")

class QASystem:
    def __init__(self, model_name="llama3", use_faiss=True, target_chunk_tokens=512, use_hybrid=True, retrieval_mode='hybrid'):
        """
        Initialize the Q&A system with embedding model and LLM

        Args:
            model_name: LLM model name
            use_faiss: Whether to use FAISS index (deprecated, use retrieval_mode instead)
            target_chunk_tokens: Target tokens per chunk
            use_hybrid: Whether to use hybrid search (deprecated, use retrieval_mode instead)
            retrieval_mode: 'faiss' (dense only), 'bm25' (sparse only), or 'hybrid' (both)
        """
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm_model = model_name
        self.documents = {}  # Store documents by filename
        self.embeddings = {}  # Store embeddings by filename (for backward compatibility)
        self.chunks = {}  # Store text chunks by filename

        # Token-aware chunking parameters
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except:
            # Fallback to approximate token counting if tiktoken fails
            self.tokenizer = None

        self.target_chunk_tokens = target_chunk_tokens  # Optimal for all-MiniLM-L6-v2
        self.min_chunk_tokens = 128  # Avoid tiny chunks
        self.max_chunk_tokens = 768  # Model limit consideration

        # Set retrieval mode (new unified approach)
        # Only use backward compatibility if retrieval_mode is default 'hybrid'
        if retrieval_mode == 'hybrid':
            # Backward compatibility with old parameters
            if not use_faiss and use_hybrid:
                self.retrieval_mode = 'bm25'
            elif use_faiss and not use_hybrid:
                self.retrieval_mode = 'faiss'
            else:
                self.retrieval_mode = 'hybrid'
        else:
            # Explicit retrieval_mode takes precedence
            self.retrieval_mode = retrieval_mode.lower()

        # FAISS setup (for dense retrieval)
        self.use_faiss = self.retrieval_mode in ['faiss', 'hybrid']
        self.embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
        self.faiss_index = None
        self.chunk_metadata = []  # Store metadata for each chunk in FAISS

        # BM25 setup (for sparse retrieval)
        self.use_hybrid = self.retrieval_mode in ['bm25', 'hybrid']
        self.bm25_index = None
        self.tokenized_corpus = []  # Tokenized chunks for BM25
        self.bm25_doc_mapping = []  # Maps BM25 index to chunk metadata

        if self.use_faiss:
            # Initialize FAISS index with cosine similarity
            # Using IndexFlatIP for inner product (we'll normalize vectors for cosine similarity)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            print(f"✅ FAISS index initialized (mode: {self.retrieval_mode})")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text
    
    def extract_pdf_with_structure(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], str]:
        """Extract text from PDF with paragraph and table structure using PyMuPDF"""
        chunks = []
        full_text = ""
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc, 1):
                # Extract text blocks (paragraphs)
                blocks = page.get_text("blocks")
                
                # Extract tables
                tables = page.find_tables()
                table_areas = []
                
                # Process tables first and mark their areas
                for table_idx, table in enumerate(tables):
                    if table:
                        # Get table bbox to avoid duplicate text extraction
                        bbox = table.bbox
                        table_areas.append(bbox)
                        
                        # Extract table data
                        table_data = table.extract()
                        if table_data:
                            # Convert table to text format
                            table_text = "\n".join(
                                " | ".join(str(cell) if cell else "" for cell in row)
                                for row in table_data if row
                            )
                            
                            if table_text.strip():
                                chunks.append({
                                    "type": "table",
                                    "content": table_text,
                                    "page": page_num,
                                    "metadata": f"Table on page {page_num}"
                                })
                                full_text += f"\n[TABLE]\n{table_text}\n[/TABLE]\n"
                
                # Process text blocks (paragraphs)
                for block in blocks:
                    # block is (x0, y0, x1, y1, "text", block_no, block_type)
                    if len(block) >= 5:
                        x0, y0, x1, y1 = block[:4]
                        text = block[4]
                        
                        # Check if this block overlaps with any table
                        is_in_table = False
                        for table_bbox in table_areas:
                            if self._bbox_overlap((x0, y0, x1, y1), table_bbox):
                                is_in_table = True
                                break
                        
                        # Only process if not part of a table and has meaningful text
                        if not is_in_table and isinstance(text, str) and text.strip():
                            # Clean the text
                            text = text.strip()
                            
                            # Skip very short blocks (likely headers/footers)
                            if len(text) > 20:
                                chunks.append({
                                    "type": "paragraph",
                                    "content": text,
                                    "page": page_num,
                                    "metadata": f"Paragraph on page {page_num}"
                                })
                                full_text += text + "\n\n"
            
            doc.close()
            
            # Merge small adjacent paragraphs if needed
            merged_chunks = self._merge_small_chunks(chunks)
            
            return merged_chunks, full_text
            
        except Exception as e:
            print(f"Error with structured extraction: {e}")
            # Fallback to simple text extraction with same library
            try:
                text = self.extract_text_from_pdf(pdf_path)
                return [], text
            except:
                print(f"Could not extract text from {pdf_path}")
                return [], ""
    
    def _bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Check if two bounding boxes overlap"""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        
        # Check if one rectangle is to the left of the other
        if x1_1 < x0_2 or x1_2 < x0_1:
            return False
        # Check if one rectangle is above the other
        if y1_1 < y0_2 or y1_2 < y0_1:
            return False
        return True
    
    def _merge_small_chunks(self, chunks: List[Dict[str, Any]], min_size: int = 100) -> List[Dict[str, Any]]:
        """Merge very small paragraphs with adjacent ones"""
        if not chunks:
            return chunks
        
        merged = []
        current = None
        
        for chunk in chunks:
            if chunk["type"] == "table":
                # Never merge tables
                if current:
                    merged.append(current)
                    current = None
                merged.append(chunk)
            elif chunk["type"] == "paragraph":
                if current is None:
                    current = chunk
                elif len(chunk["content"]) < min_size and current["page"] == chunk["page"]:
                    # Merge small chunk with current
                    current["content"] += "\n\n" + chunk["content"]
                elif len(current["content"]) < min_size and current["page"] == chunk["page"]:
                    # Current is small, merge with this chunk
                    current["content"] += "\n\n" + chunk["content"]
                else:
                    # Both are large enough, keep separate
                    merged.append(current)
                    current = chunk
        
        if current:
            merged.append(current)
        
        return merged
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback approximation"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximation: ~4 characters per token for English text
            return len(text) // 4

    def chunk_text(self, text: str) -> List[str]:
        """Simple working chunking - splits text into fixed-size chunks"""
        chunks = []
        chunk_size = 2000  # Characters per chunk

        # Clean the text
        text = text.replace('\r\n', '\n').strip()

        # If text is empty, return empty list
        if not text:
            return []

        # Split into chunks of fixed size
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            # Only add non-empty chunks with reasonable content
            if len(chunk.strip()) > 100:
                chunks.append(chunk)

        # If no chunks created but we have text, return the whole text
        if not chunks and len(text) > 0:
            chunks = [text]

        return chunks

    def academic_tokenizer(self, text: str) -> List[str]:
        """Tokenize text for BM25, preserving academic terms and references"""
        # Preserve special patterns as single tokens
        text = re.sub(r'(Table|Figure|Section|Equation|Appendix)\s+(\d+)', r'\1_\2', text)
        text = re.sub(r'et\s+al\.', 'et_al', text)

        # Basic word tokenization
        tokens = text.lower().split()

        # Remove very short tokens and punctuation-only tokens
        tokens = [t for t in tokens if len(t) > 1 and not all(c in '.,;:!?()[]{}' for c in t)]

        return tokens

    def add_document(self, filename: str, pdf_path: str = None, text: str = None):
        """Add a document to the Q&A system"""
        chunks_to_embed = []
        
        # Try structured extraction first if PDF path provided
        if pdf_path and os.path.exists(pdf_path):
            structured_chunks, full_text = self.extract_pdf_with_structure(pdf_path)
            
            if structured_chunks:
                # Use structured chunks
                print(f"Using structured extraction: {len(structured_chunks)} chunks")
                print(f"  - Tables: {sum(1 for c in structured_chunks if c['type'] == 'table')}")
                print(f"  - Paragraphs: {sum(1 for c in structured_chunks if c['type'] == 'paragraph')}")
                
                # Store full text
                self.documents[filename] = full_text
                
                # Process structured chunks with smart chunking
                all_content = '\n\n'.join([chunk["content"] for chunk in structured_chunks])
                chunks_to_embed = self.chunk_text(all_content)

                # Store chunks with metadata
                self.chunks[filename] = chunks_to_embed
                print(f"  → Created {len(chunks_to_embed)} token-aware chunks")
                print(f"  → Avg tokens/chunk: {np.mean([self.count_tokens(c) for c in chunks_to_embed]):.0f}")
                
            else:
                # Fallback to simple extraction (still using PyMuPDF)
                text = self.extract_text_from_pdf(pdf_path)
                text = self.clean_text(text)
                self.documents[filename] = text
                chunks_to_embed = self.chunk_text(text)
                self.chunks[filename] = chunks_to_embed
                print(f"Created {len(chunks_to_embed)} chunks (avg {np.mean([self.count_tokens(c) for c in chunks_to_embed]):.0f} tokens)")
                
        elif text:
            # Use provided text
            text = self.clean_text(text)
            self.documents[filename] = text
            chunks_to_embed = self.chunk_text(text)
            self.chunks[filename] = chunks_to_embed
        else:
            print(f"No text or valid PDF path provided for {filename}")
            return False
        
        # Generate embeddings for chunks
        if chunks_to_embed:
            # Build BM25 index if needed (for BM25-only or Hybrid modes)
            if self.use_hybrid:
                self._update_bm25_index(chunks_to_embed, filename)

            # Generate embeddings if FAISS is used
            if self.use_faiss or not self.use_hybrid:
                embeddings = self.embedding_model.encode(chunks_to_embed)

            if self.use_faiss:
                # Normalize embeddings for cosine similarity
                embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                # Add to FAISS index
                self.faiss_index.add(embeddings_normalized.astype('float32'))

                # Store metadata for each chunk
                for i, chunk in enumerate(chunks_to_embed):
                    self.chunk_metadata.append({
                        'filename': filename,
                        'chunk_idx': i,
                        'content': chunk
                    })
                print(f"Added {filename} with {len(chunks_to_embed)} chunks to FAISS index")
            elif not self.use_faiss and not self.use_hybrid:
                # Original implementation for backward compatibility
                self.embeddings[filename] = embeddings
                print(f"Added {filename} with {len(chunks_to_embed)} chunks")

            return True
        return False

    def _update_bm25_index(self, new_chunks: List[str], filename: str):
        """Update BM25 index with new chunks"""
        # Tokenize new chunks
        new_tokenized = [self.academic_tokenizer(chunk) for chunk in new_chunks]

        # Add to corpus
        for i, (chunk, tokens) in enumerate(zip(new_chunks, new_tokenized)):
            self.tokenized_corpus.append(tokens)
            self.bm25_doc_mapping.append({
                'filename': filename,
                'chunk_idx': i,
                'content': chunk
            })

        # Rebuild BM25 index with all documents
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        print(f"  Updated BM25 index - total docs: {len(self.tokenized_corpus)}")

    def clean_text(self, text: str) -> str:
        """Clean text for better processing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep more characters including numbers, periods, parentheses, etc.
        # Only remove truly problematic characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()

    def calculate_dynamic_threshold(self, scores: np.ndarray, min_threshold: float = 0.3, max_threshold: float = 0.9) -> float:
        """Calculate dynamic threshold based on score distribution"""
        if len(scores) == 0:
            return min_threshold

        # Calculate statistics
        mean = np.mean(scores)
        std = np.std(scores)

        # Adaptive threshold logic
        if std > 0.15:  # Diverse scores - there are clear winners
            threshold = mean + std  # One standard deviation above mean
        else:  # Similar scores - be more selective
            threshold = np.percentile(scores, 80)  # Top 20%

        # Apply bounds to prevent extreme values
        final_threshold = np.clip(threshold, min_threshold, max_threshold)

        print(f"[THRESHOLD CALCULATION] Mean: {mean:.3f}, Std: {std:.3f}, Threshold: {final_threshold:.3f}")
        return final_threshold

    def apply_threshold_with_minimum(self, chunks_with_scores: List[Dict[str, Any]], threshold: float, min_chunks: int = 3) -> List[Dict[str, Any]]:
        """Apply threshold while ensuring minimum chunks are returned"""
        # Filter by threshold
        above_threshold = [chunk for chunk in chunks_with_scores if chunk['score'] > threshold]

        if len(above_threshold) >= min_chunks:
            return above_threshold

        # If we don't have enough chunks above threshold, take the top min_chunks
        chunks_sorted = sorted(chunks_with_scores, key=lambda x: x['score'], reverse=True)
        return chunks_sorted[:min_chunks]
    
    def find_relevant_chunks(self, query: str, top_k: int = 10, hybrid_alpha: float = 0.5, verbose: bool = True) -> List[Dict[str, Any]]:
        """Find the most relevant text chunks for a query using FAISS or fallback"""
        # Log query with full details
        print(f"\n{'='*80}")
        print(f"[QUERY RECEIVED] at {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        print(f"\n[FULL QUERY]:")
        print(f"  \"{query}\"")
        print(f"\n[QUERY DETAILS]:")
        print(f"  - Length: {len(query)} chars")
        print(f"  - Tokens: ~{self.count_tokens(query)}")
        print(f"  - Searching for top {top_k} chunks")
        print(f"  - Verbose mode: {verbose}")
        if self.use_hybrid and self.bm25_index is not None:
            print(f"  - Hybrid search: ENABLED (alpha={hybrid_alpha})")
            print(f"  - FAISS weight: {hybrid_alpha}")
            print(f"  - BM25 weight: {1-hybrid_alpha}")
        else:
            print(f"  - Hybrid search: DISABLED (using FAISS only)")
        print(f"{'='*80}\n")

        # Handle different retrieval modes
        if self.retrieval_mode == 'bm25':
            # BM25-only retrieval
            if not self.bm25_index or len(self.tokenized_corpus) == 0:
                print("⚠️ BM25 index not available")
                return []

            print(f"[BM25-ONLY SEARCH] Retrieving top {top_k} chunks")
            query_tokens = self.academic_tokenizer(query)
            bm25_scores = self.bm25_index.get_scores(query_tokens)

            # Get top BM25 results
            bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

            all_relevant = []
            for idx in bm25_top_indices:
                if idx < len(self.bm25_doc_mapping) and bm25_scores[idx] > 0.01:
                    metadata = self.bm25_doc_mapping[idx]
                    all_relevant.append({
                        'filename': metadata['filename'],
                        'chunk': metadata['content'],
                        'score': float(bm25_scores[idx]),
                        'retrieval_method': 'BM25'
                    })

            print(f"  Retrieved {len(all_relevant)} chunks via BM25")

            # Log and return BM25 results
            if verbose:
                print(f"\n[RETRIEVED CHUNKS] Found {len(all_relevant)} relevant chunks")
                for i, chunk in enumerate(all_relevant, 1):
                    print(f"\n{i}. {chunk['filename']} (Score: {chunk['score']:.3f})")
                    print(f"   {chunk['chunk'][:100]}...")
            else:
                print(f"\n[RETRIEVED CHUNKS] Found {len(all_relevant)} relevant chunks")

            return all_relevant

        elif self.use_faiss and self.faiss_index and self.faiss_index.ntotal > 0:
            # FAISS or Hybrid search
            query_embedding = self.embedding_model.encode([query])
            query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)

            # Search in FAISS index - get more candidates initially
            search_k = min(top_k * 3, self.faiss_index.ntotal)  # Get 3x candidates for better threshold calculation
            scores, indices = self.faiss_index.search(query_embedding_normalized.astype('float32'), search_k)

            # Collect all valid chunks with scores
            chunks_with_scores = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunk_metadata):  # Valid index
                    metadata = self.chunk_metadata[idx]
                    chunks_with_scores.append({
                        'filename': metadata['filename'],
                        'chunk': metadata['content'],
                        'score': float(score)
                    })

            # Calculate dynamic threshold based on score distribution
            if chunks_with_scores:
                all_scores = np.array([c['score'] for c in chunks_with_scores])
                threshold = self.calculate_dynamic_threshold(all_scores)

                # Apply threshold with minimum guarantee
                all_relevant = self.apply_threshold_with_minimum(chunks_with_scores, threshold, min_chunks=3)

                # If hybrid search is enabled, combine with BM25 results
                if self.retrieval_mode == 'hybrid' and self.bm25_index is not None:
                    all_relevant = self._hybrid_search(query, all_relevant, top_k, hybrid_alpha)
                else:
                    # Limit to top_k for FAISS-only
                    all_relevant = all_relevant[:top_k]
            else:
                all_relevant = []

            # Log retrieved chunks with full content if verbose
            if verbose:
                print(f"\n[RETRIEVED CHUNKS] Found {len(all_relevant)} relevant chunks")
                print(f"{'-'*80}")

                # Show ALL chunks with FULL content
                for i, chunk in enumerate(all_relevant, 1):
                    print(f"\n{'='*40}")
                    print(f"CHUNK {i}/{len(all_relevant)}")
                    print(f"{'='*40}")
                    print(f"Source: {chunk['filename']}")
                    print(f"Score: {chunk['score']:.3f}")
                    if 'retrieval_method' in chunk:
                        print(f"Method: {chunk['retrieval_method']}")
                    print(f"\n[FULL CONTENT]:")
                    print(chunk['chunk'])  # Full content, no truncation
                    print(f"\n[END OF CHUNK {i}]")

                print(f"\n{'-'*80}")
                print(f"TOTAL CHUNKS RETRIEVED: {len(all_relevant)}")
                print(f"{'-'*80}\n")
            else:
                # Compact logging
                print(f"\n[RETRIEVED CHUNKS] Found {len(all_relevant)} relevant chunks")

            return all_relevant
        
        elif not self.use_faiss and self.embeddings:
            # Original implementation for backward compatibility
            query_embedding = self.embedding_model.encode([query])[0]
            all_relevant = []
            
            for filename, embeddings in self.embeddings.items():
                similarities = cosine_similarity([query_embedding], embeddings)[0]
                # Get more candidates for threshold calculation
                search_k = min(top_k * 3, len(similarities))
                top_indices = np.argsort(similarities)[-search_k:][::-1]

                for idx in top_indices:
                    all_relevant.append({
                        'filename': filename,
                        'chunk': self.chunks[filename][idx],
                        'score': float(similarities[idx])
                    })
            
            # Sort all results by score
            all_relevant.sort(key=lambda x: x['score'], reverse=True)

            # Calculate dynamic threshold
            if all_relevant:
                all_scores = np.array([c['score'] for c in all_relevant])
                threshold = self.calculate_dynamic_threshold(all_scores)

                # Apply threshold with minimum guarantee
                result = self.apply_threshold_with_minimum(all_relevant, threshold, min_chunks=3)

                # If hybrid search is enabled, combine with BM25 results
                if self.use_hybrid and self.bm25_index is not None:
                    result = self._hybrid_search(query, result, top_k, hybrid_alpha)
                else:
                    # Limit to top_k for dense-only
                    result = result[:top_k]
            else:
                result = []

            # Log retrieved chunks with full content if verbose
            if verbose:
                print(f"\n[RETRIEVED CHUNKS] Found {len(result)} relevant chunks")
                print(f"{'-'*80}")

                # Show ALL chunks with FULL content
                for i, chunk in enumerate(result, 1):
                    print(f"\n{'='*40}")
                    print(f"CHUNK {i}/{len(result)}")
                    print(f"{'='*40}")
                    print(f"Source: {chunk['filename']}")
                    print(f"Score: {chunk['score']:.3f}")
                    if 'retrieval_method' in chunk:
                        print(f"Method: {chunk['retrieval_method']}")
                    print(f"\n[FULL CONTENT]:")
                    print(chunk['chunk'])  # Full content, no truncation
                    print(f"\n[END OF CHUNK {i}]")

                print(f"\n{'-'*80}")
                print(f"TOTAL CHUNKS RETRIEVED: {len(result)}")
                print(f"{'-'*80}\n")
            else:
                # Compact logging
                print(f"\n[RETRIEVED CHUNKS] Found {len(result)} relevant chunks")

            return result
        
        else:
            return []

    def _hybrid_search(self, query: str, dense_results: List[Dict[str, Any]], top_k: int, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Combine dense (FAISS) and sparse (BM25) search results using reciprocal rank fusion"""
        if not self.bm25_index or len(self.tokenized_corpus) == 0:
            return dense_results[:top_k]

        # Get BM25 results
        print(f"[HYBRID SEARCH] Combining FAISS and BM25 results (alpha={alpha})")

        # Tokenize query for BM25
        query_tokens = self.academic_tokenizer(query)

        # Get BM25 scores for all documents
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Debug logging
        print(f"  Query tokens: {query_tokens[:5]}...")  # First 5 tokens
        print(f"  BM25 corpus size: {len(self.tokenized_corpus)}")
        print(f"  BM25 scores: max={max(bm25_scores) if len(bm25_scores) > 0 else 0:.3f}, "
              f"min={min(bm25_scores) if len(bm25_scores) > 0 else 0:.3f}")

        # Get top BM25 results
        bm25_top_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]  # Get 2x candidates

        bm25_results = []
        for idx in bm25_top_indices:
            # Lower threshold for single/few documents
            min_bm25_score = 0.01 if len(self.tokenized_corpus) < 5 else 0.1
            if idx < len(self.bm25_doc_mapping) and bm25_scores[idx] > min_bm25_score:
                metadata = self.bm25_doc_mapping[idx]
                bm25_results.append({
                    'filename': metadata['filename'],
                    'chunk': metadata['content'],
                    'score': float(bm25_scores[idx]),
                    'bm25_rank': len(bm25_results)
                })

        # Apply reciprocal rank fusion
        combined_results = self._reciprocal_rank_fusion(dense_results, bm25_results, alpha)

        print(f"  Dense results: {len(dense_results)}, BM25 results: {len(bm25_results)}")
        print(f"  Combined unique results: {len(combined_results)}")

        return combined_results[:top_k]

    def _reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], alpha: float = 0.5, k: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion to combine dense and sparse results"""
        # Create a scoring dictionary
        rrf_scores = {}

        # Add dense retrieval scores
        for rank, result in enumerate(dense_results):
            # Create unique key for each chunk
            key = (result['filename'], result['chunk'][:100])  # Use first 100 chars as key

            # RRF formula weighted by alpha
            score = alpha * (1.0 / (k + rank + 1))

            if key not in rrf_scores:
                rrf_scores[key] = {'result': result, 'score': 0}
            rrf_scores[key]['score'] += score
            rrf_scores[key]['result']['dense_rank'] = rank

        # Add sparse retrieval scores
        for rank, result in enumerate(sparse_results):
            key = (result['filename'], result['chunk'][:100])

            # RRF formula weighted by (1-alpha)
            score = (1 - alpha) * (1.0 / (k + rank + 1))

            if key not in rrf_scores:
                rrf_scores[key] = {'result': result, 'score': 0}
            rrf_scores[key]['score'] += score
            rrf_scores[key]['result']['sparse_rank'] = rank

        # Sort by combined RRF score
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x['score'], reverse=True)

        # Extract results and update scores
        final_results = []
        for item in sorted_results:
            result = item['result'].copy()
            result['hybrid_score'] = item['score']
            result['retrieval_method'] = 'dense' if 'dense_rank' in result else 'sparse'
            if 'dense_rank' in result and 'sparse_rank' in result:
                result['retrieval_method'] = 'both'
            final_results.append(result)

        return final_results
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using Ollama based on relevant chunks"""
        if not relevant_chunks:
            no_info_msg = "I couldn't find relevant information in the uploaded documents to answer your question."
            print(f"\n[NO CHUNKS FOUND] {no_info_msg}\n")
            return no_info_msg
        
        # Prepare context from relevant chunks
        context = "\n\n".join([
            f"From {chunk['filename']} (relevance: {chunk['score']:.2f}):\n{chunk['chunk']}"
            for chunk in relevant_chunks
        ])

        # Log context being sent to LLM with FULL context
        print(f"\n[CONTEXT PREPARED FOR LLM]")
        print(f"{'-'*80}")
        print(f"Total context length: {len(context)} chars (~{self.count_tokens(context)} tokens)")
        print(f"Number of chunks used: {len(relevant_chunks)}")
        print(f"Files referenced: {', '.join(set(c['filename'] for c in relevant_chunks))}")
        print(f"\n[FULL CONTEXT BEING SENT TO LLM]:")
        print(f"{'-'*40}")
        print(context)  # Show full context
        print(f"{'-'*40}")
        print(f"{'-'*80}\n")
        
        # Check if this is a dataset-related query
        dataset_keywords = ['dataset', 'data source', 'data from', 'database', 'repository', 
                           'observations', 'measurements', 'records', 'ERA5', 'MODIS', 'NSIDC']
        is_dataset_query = any(keyword.lower() in query.lower() for keyword in dataset_keywords)
        
        if is_dataset_query:
            # Use strict dataset extraction prompt
            prompt = f"""You are a dataset information specialist analyzing research documents.

RULES:
- ALWAYS use EXACT dataset names (e.g., ERA5, MODIS, NSIDC-0051)
- NEVER say "various datasets" or "climate data" - be specific
- Include time periods and locations when mentioned in the context
- If no specific datasets found in the context, explicitly say "No specific datasets mentioned"
- List each dataset with its full name, time period, and geographic coverage if available

Context from documents:
{context}

User Question: {query}

Based ONLY on the information in the context above, provide specific dataset information:"""
        else:
            # Use regular prompt for non-dataset queries
            prompt = f"""You are a helpful assistant analyzing research documents. 
Based on the following context from the uploaded PDFs, please answer the user's question.
If the answer is not in the context, say so clearly.

Context from documents:
{context}

User Question: {query}

Please provide a clear, concise answer based on the information provided in the context:"""

        try:
            # Log LLM request with FULL prompt
            print(f"\n[GENERATING ANSWER] Using model: {self.llm_model}")
            print(f"Prompt length: {len(prompt)} chars")
            print(f"\n[FULL PROMPT TO LLM]:")
            print(f"{'-'*40}")
            print(prompt)  # Show full prompt
            print(f"{'-'*40}\n")

            # Generate response using Ollama
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response['message']['content']

            # Log generated answer with full text
            print(f"\n[ANSWER GENERATED - FULL RESPONSE]")
            print(f"{'-'*80}")
            print(answer)  # Already shows full answer
            print(f"{'-'*80}")
            print(f"\nAnswer length: {len(answer)} chars (~{self.count_tokens(answer)} tokens)")
            
            # Add source citations
            sources = list(set([chunk['filename'] for chunk in relevant_chunks]))
            if sources:
                answer += f"\n\nSources: {', '.join(sources)}"
                print(f"\nSources cited: {', '.join(sources)}")

            print(f"\n{'='*80}")
            print(f"[Q&A COMPLETED] at {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*80}\n")

            return answer
            
        except Exception as e:
            print(f"\n[ERROR] Failed to generate answer: {e}")
            print(f"{'='*80}\n")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, query: str, verbose: bool = True) -> str:  # Changed default to True for logging
        """Main method to answer a question"""
        if not self.documents:
            no_docs_msg = "No documents have been uploaded yet. Please upload PDFs first."
            print(f"\n[WARNING] {no_docs_msg}\n")
            return no_docs_msg

        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(query, verbose=verbose)

        # Generate answer
        answer = self.generate_answer(query, relevant_chunks)
        return answer
    
    def get_document_summary(self, filename: str) -> str:
        """Generate a summary of a specific document"""
        if filename not in self.documents:
            return f"Document {filename} not found"
        
        text = self.documents[filename]
        # Take first 2000 characters for summary
        text_sample = text[:2000] if len(text) > 2000 else text
        
        prompt = f"""Please provide a brief summary of this document in 2-3 sentences:

{text_sample}

Summary:"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"Could not generate summary: {str(e)}"
    
    def list_documents(self) -> List[str]:
        """List all loaded documents"""
        return list(self.documents.keys())
    
    def clear_documents(self):
        """Clear all loaded documents and FAISS index"""
        self.documents.clear()
        self.embeddings.clear()
        self.chunks.clear()
        
        if self.use_faiss:
            # Reset FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_metadata = []
            print("✅ FAISS index cleared")
    
    def reset_and_reload(self):
        """Reset the Q&A system for fresh loading"""
        self.clear_documents()
        print("Q&A system reset - ready for new documents")
    
    def save_faiss_index(self, path: str = "faiss_index.pkl"):
        """Save FAISS index and metadata to disk"""
        if self.use_faiss and self.faiss_index:
            # Save both index and metadata
            with open(path, 'wb') as f:
                pickle.dump({
                    'index': faiss.serialize_index(self.faiss_index),
                    'metadata': self.chunk_metadata,
                    'documents': self.documents,
                    'chunks': self.chunks
                }, f)
            print(f"✅ FAISS index saved to {path}")
    
    def load_faiss_index(self, path: str = "faiss_index.pkl"):
        """Load FAISS index and metadata from disk"""
        if self.use_faiss and os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.faiss_index = faiss.deserialize_index(data['index'])
                self.chunk_metadata = data['metadata']
                self.documents = data['documents']
                self.chunks = data['chunks']
            print(f"✅ FAISS index loaded from {path}")
            print(f"   - Total chunks in index: {self.faiss_index.ntotal}")
            print(f"   - Documents: {list(self.documents.keys())}")
            return True
        return False

# Singleton instance for the application
qa_system = QASystem()