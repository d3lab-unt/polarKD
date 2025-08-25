"""
Q&A Module for PDF Knowledge Explorer
Implements RAG (Retrieval Augmented Generation) using uploaded PDFs
"""

import os
import numpy as np
from typing import List, Dict, Any
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
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
        """Extract text from PDF file"""
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
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better context"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def add_document(self, filename: str, pdf_path: str = None, text: str = None):
        """Add a document to the Q&A system"""
        # Extract text if PDF path provided
        if pdf_path and os.path.exists(pdf_path):
            text = self.extract_text_from_pdf(pdf_path)
        elif not text:
            print(f"No text or valid PDF path provided for {filename}")
            return False
        
        # Clean the text
        text = self.clean_text(text)
        
        # Store the full document
        self.documents[filename] = text
        
        # Create chunks
        chunks = self.chunk_text(text)
        self.chunks[filename] = chunks
        
        # Generate embeddings for chunks
        if chunks:
            embeddings = self.embedding_model.encode(chunks)
            self.embeddings[filename] = embeddings
            print(f"Added {filename} with {len(chunks)} chunks")
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
    
    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
        
        # Create prompt
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