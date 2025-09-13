"""
Q&A Module for PDF Knowledge Explorer
Implements RAG (Retrieval Augmented Generation) using uploaded PDFs
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import fitz  # PyMuPDF
import re
from collections import defaultdict

class QASystem:
    def __init__(self, model_name="llama3"):
        """Initialize the Q&A system with embedding model and LLM"""
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm_model = model_name
        self.documents = {}  # Store documents by filename
        self.embeddings = {}  # Store embeddings by filename
        self.chunks = {}  # Store text chunks by filename
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using pdfplumber (fallback method)"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
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
            print(f"Error with PyMuPDF extraction: {e}")
            print("Falling back to pdfplumber...")
            # Fallback to original method
            text = self.extract_text_from_pdf(pdf_path)
            return [], text
    
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
    
    def chunk_text(self, text: str, chunk_size: int = 3000, overlap: int = 500) -> List[str]:
        """Split text into overlapping chunks for better context"""
        # Use character-based chunking with sentence boundaries
        sentences = text.replace('. ', '.<<<SPLIT>>>').split('<<<SPLIT>>>')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk_size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
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
                
                # Extract content from structured chunks for embedding
                chunks_to_embed = [chunk["content"] for chunk in structured_chunks]
                
                # Store chunks with metadata
                self.chunks[filename] = chunks_to_embed
                
            else:
                # Fallback to regular extraction
                text = self.extract_text_from_pdf(pdf_path)
                text = self.clean_text(text)
                self.documents[filename] = text
                chunks_to_embed = self.chunk_text(text)
                self.chunks[filename] = chunks_to_embed
                
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
            embeddings = self.embedding_model.encode(chunks_to_embed)
            self.embeddings[filename] = embeddings
            print(f"Added {filename} with {len(chunks_to_embed)} chunks")
            return True
        return False
    
    def clean_text(self, text: str) -> str:
        """Clean text for better processing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep more characters including numbers, periods, parentheses, etc.
        # Only remove truly problematic characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()
    
    def find_relevant_chunks(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find the most relevant text chunks for a query"""
        if not self.embeddings:
            return []
        
        # Encode the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Find relevant chunks from all documents
        all_relevant = []
        
        for filename, embeddings in self.embeddings.items():
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # Get top chunks from this document
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            for idx in top_indices:
                if similarities[idx] > 0.15:  # Lowered threshold for better recall
                    all_relevant.append({
                        'filename': filename,
                        'chunk': self.chunks[filename][idx],
                        'score': float(similarities[idx])
                    })
        
        # Sort by score and return top_k overall
        all_relevant.sort(key=lambda x: x['score'], reverse=True)
        return all_relevant[:top_k]
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using Ollama based on relevant chunks"""
        if not relevant_chunks:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
        
        # Prepare context from relevant chunks
        context = "\n\n".join([
            f"From {chunk['filename']} (relevance: {chunk['score']:.2f}):\n{chunk['chunk']}"
            for chunk in relevant_chunks
        ])
        
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
            # Generate response using Ollama
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response['message']['content']
            
            # Add source citations
            sources = list(set([chunk['filename'] for chunk in relevant_chunks]))
            if sources:
                answer += f"\n\n📚 Sources: {', '.join(sources)}"
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, query: str, verbose: bool = False) -> str:
        """Main method to answer a question"""
        if not self.documents:
            return "No documents have been uploaded yet. Please upload PDFs first."
        
        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(query)
        
        if verbose:
            print(f"Found {len(relevant_chunks)} relevant chunks")
            for chunk in relevant_chunks:
                print(f"- {chunk['filename']}: {chunk['score']:.3f}")
        
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
        """Clear all loaded documents"""
        self.documents.clear()
        self.embeddings.clear()
        self.chunks.clear()
    
    def reset_and_reload(self):
        """Reset the Q&A system for fresh loading"""
        self.clear_documents()
        print("Q&A system reset - ready for new documents")

# Singleton instance for the application
qa_system = QASystem()