#!/usr/bin/env python3
"""
Task 1: Data Ingestion
- Extract text from PDF files using various extraction methods
- Clean and normalize extracted text with source markers
- Generate manifest file for downstream processing
- Support multiple PDF processing strategies (native, OCR fallback)

Environment Variables:
- DATA_DIR: Input directory containing PDF files
- CACHE_DIR: Output directory for cleaned text files

Requirements: pip install -r requirements.txt
"""

import logging
import os
import re
from pathlib import Path

import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from environment variables"""
    config = {
        'data_dir': Path(os.getenv('DATA_DIR', '../data/sample')),
        'cache_dir': Path(os.getenv('CACHE_DIR', '../cleaned')),
    }
    
    # Create directories if they don't exist
    for path in config.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    
    return config

def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    # Remove multiple spaces before newlines
    text = re.sub(r"\s+\n", "\n", text)
    # Replace single newlines with spaces (preserve paragraph breaks)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Remove hyphen line breaks
    text = re.sub(r"-\s*\n", "", text)
    # Remove page numbers and pagination
    text = re.sub(r"Page \d+|\d+\s*/\s*\d+", "", text, flags=re.I)
    # Normalize spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def extract_pdf_to_txt(pdf_path: Path, cache_dir: Path) -> Path:
    """Extract text from PDF and save to cache directory"""
    out_txt = cache_dir / (pdf_path.stem + ".txt")
    
    if out_txt.exists():
        logger.info(f"Text file already exists: {out_txt}")
        return out_txt
    
    logger.info(f"Extracting text from: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        parts = []
        
        for i, page in enumerate(doc):
            raw = page.get_text("text") or ""
            cleaned = clean_text(raw)
            parts.append(f"[SOURCE=native PAGE={i+1} FILE={pdf_path.name}]\n{cleaned}\n")
        
        all_text = "\n\n".join(parts)
        out_txt.write_text(all_text, encoding="utf-8")
        logger.info(f"Saved cleaned text to: {out_txt}")
        
        doc.close()
        return out_txt
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        raise

def main():
    """Main function to process all PDFs in data directory"""
    config = load_config()
    data_dir = config['data_dir']
    cache_dir = config['cache_dir']
    
    # Find all PDF files
    pdfs = list(data_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(pdfs)} PDF files")
    
    if not pdfs:
        logger.warning("No PDF files found in data directory")
        return
    
    # Process each PDF
    processed_files = []
    for pdf_path in pdfs:
        try:
            txt_path = extract_pdf_to_txt(pdf_path, cache_dir)
            processed_files.append(txt_path)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
    
    logger.info(f"Successfully processed {len(processed_files)} files")
    
    # Save manifest for next task
    manifest_path = cache_dir / "manifest.txt"
    with open(manifest_path, 'w') as f:
        for txt_path in processed_files:
            f.write(f"{txt_path}\n")
    
    logger.info(f"Saved file manifest to: {manifest_path}")

if __name__ == "__main__":
    main()