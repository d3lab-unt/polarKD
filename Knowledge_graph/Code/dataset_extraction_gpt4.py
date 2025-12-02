"""
GPT-4o-mini based dataset extraction for research papers.
Extracts ALL datasets and labels them as PRIMARY or CITED.
"""

import os
import json
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in parent directory (Knowledge_graph/.env)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from openai import OpenAI

@dataclass
class DatasetMetadata:
    """Metadata for a dataset mentioned in a paper."""
    source: str
    variables: List[str]
    time_period: str
    location: str
    context: str
    chunk_indices: List[int]
    confidence_score: float
    dataset_type: str  # "primary" or "cited"
    usage_description: str
    citation_info: Optional[str] = None

class GPT4DatasetExtractor:
    """Extract ALL datasets from research papers with PRIMARY/CITED labels."""
    
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        
    def _create_extraction_prompt(self) -> str:
        """Create the system prompt for dataset extraction."""
        return """You are a scientific dataset extraction expert. Extract ALL datasets mentioned in research papers and classify each as PRIMARY or CITED.

CRITICAL: Extract EVERY dataset, model, reanalysis, observation system, or data source mentioned.

CLASSIFICATION:

PRIMARY - Datasets the AUTHORS of THIS paper created/collected/generated:
  - Measurements THEY collected: "We deployed", "Our instruments measured", "We observed"
  - Simulations THEY ran: "We ran CCSM4", "Our model experiments", "We simulated using"
  - Analysis THEY performed: "We analyzed and computed", "Our calculations show"
  - Model output THEY generated: "Our simulations produced", "Model results from our experiments"

CITED - Datasets from OTHER sources that they reference/use:
  - Background references: "According to NASA", "PIOMAS shows", "Previous studies"
  - Input data: "Forced by ERA5", "Using TOPAZ", "Initial conditions from"
  - Comparison data: "Validated against NSIDC", "Compared with observations"
  - Any dataset they did NOT create themselves

KEY DISTINCTION:
- PRIMARY = "WE created/collected/generated this"
- CITED = "OTHERS created this, we used/referenced it"

WHAT TO EXTRACT:
✓ Climate models (CCSM4, EC-Earth, IPSL, WW3, WAM, etc.)
✓ Reanalysis products (ERA5, ERA-Interim, NCEP, JRA, etc.)
✓ Observation systems (PIOMAS, NSIDC, NASA, NOAA, etc.)
✓ Field measurements and experiments
✓ Satellite data products
✓ Model simulations and outputs
✓ ANY dataset mentioned in abstract, methods, results, discussion

Return ONLY valid JSON array (no markdown, no explanation):
[
  {
    "source": "exact dataset/model name",
    "variables": ["var1", "var2"],
    "time_period": "YYYY-YYYY or Not specified",
    "location": "location or Not specified",
    "context": "brief description",
    "confidence_score": 0.95,
    "dataset_type": "primary",
    "usage_description": "how the authors used it",
    "citation_info": "reference if cited, null if primary"
  }
]

If NO datasets in this chunk, return: []"""

    def _create_chunk_prompt(self, text_chunk: str) -> str:
        return f"""Extract ALL datasets from this text and classify as PRIMARY or CITED.

Text:
{text_chunk}

Return JSON array:"""

    def _split_into_chunks(self, text: str, chunk_size: int = 3000, 
                          overlap: int = 500) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def _extract_from_chunk(self, chunk: str, chunk_idx: int) -> Tuple[List[Dict], int]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_extraction_prompt()},
                    {"role": "user", "content": self._create_chunk_prompt(chunk)}
                ],
                temperature=0.1,
                max_tokens=2500
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            content = content.strip()
            
            if content == '[]' or not content:
                return [], response.usage.total_tokens
                
            datasets = json.loads(content)
            
            for ds in datasets:
                if 'chunk_indices' not in ds:
                    ds['chunk_indices'] = [chunk_idx]
                else:
                    ds['chunk_indices'].append(chunk_idx)
            
            return datasets, response.usage.total_tokens
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error in chunk {chunk_idx}: {e}")
            return [], 0
        except Exception as e:
            print(f"  ⚠️  Error in chunk {chunk_idx}: {e}")
            return [], 0

    def _deduplicate_datasets(self, datasets: List[Dict]) -> List[DatasetMetadata]:
        if not datasets:
            return []
        
        unique = []
        seen_sources = set()
        
        for ds in datasets:
            source_lower = ds['source'].lower().strip()
            
            is_duplicate = False
            for seen in seen_sources:
                source_words = set(source_lower.split())
                seen_words = set(seen.split())
                if not source_words or not seen_words:
                    continue
                overlap = len(source_words & seen_words)
                similarity = overlap / max(len(source_words), len(seen_words))
                if similarity > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_sources.add(source_lower)
                metadata = DatasetMetadata(
                    source=ds['source'],
                    variables=ds.get('variables', []),
                    time_period=ds.get('time_period', 'Not specified'),
                    location=ds.get('location', 'Not specified'),
                    context=ds.get('context', ''),
                    chunk_indices=ds.get('chunk_indices', []),
                    confidence_score=ds.get('confidence_score', 0.5),
                    dataset_type=ds.get('dataset_type', 'cited'),
                    usage_description=ds.get('usage_description', ''),
                    citation_info=ds.get('citation_info')
                )
                unique.append(metadata)
        
        return unique

    def extract_from_full_text(self, text: str, verbose: bool = True) -> Tuple[List[DatasetMetadata], Dict]:
        if verbose:
            print("=" * 80)
            print("GPT-4o-mini Dataset Extraction: ALL Datasets (PRIMARY/CITED)")
            print("=" * 80)
        
        start_time = time.time()
        
        chunks = self._split_into_chunks(text)
        if verbose:
            print(f"📄 Document length: {len(text):,} characters")
            print(f"🔪 Created {len(chunks)} chunks")
            print("=" * 80)
        
        all_datasets = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"Processing chunk {i+1}/{len(chunks)}...", end='\r')
            
            chunk_datasets, tokens = self._extract_from_chunk(chunk, i)
            all_datasets.extend(chunk_datasets)
            total_tokens += tokens
        
        if verbose:
            print()
        
        unique_datasets = self._deduplicate_datasets(all_datasets)
        
        primary_count = sum(1 for ds in unique_datasets if ds.dataset_type == 'primary')
        cited_count = len(unique_datasets) - primary_count
        
        input_cost = (total_tokens * 0.150) / 1_000_000
        output_cost = (total_tokens * 0.600) / 1_000_000
        total_cost = input_cost + output_cost
        
        processing_time = time.time() - start_time
        
        stats = {
            'chunks_processed': len(chunks),
            'raw_datasets': len(all_datasets),
            'unique_datasets': len(unique_datasets),
            'primary_count': primary_count,
            'cited_count': cited_count,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'processing_time': processing_time,
            'cost_per_dataset': total_cost / len(unique_datasets) if unique_datasets else 0
        }
        
        if verbose:
            print(f"✅ Processed {len(chunks)} chunks")
            print(f"📊 Found {len(all_datasets)} mentions")
            print(f"✨ Unique: {len(unique_datasets)} datasets")
            print(f"   🟢 PRIMARY: {primary_count}")
            print(f"   🔵 CITED: {cited_count}")
            print("=" * 80)
            print(f"💰 Cost: ${total_cost:.6f}")
            print(f"⏱️  Time: {processing_time:.2f}s")
            print("=" * 80)
        
        return unique_datasets, stats
