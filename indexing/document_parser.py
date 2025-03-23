import os
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Generator

from config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentParser:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_pages(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text content from each page in the PDF
        """
        pages_info = {}
        pages = []
        doc = fitz.open(file_path)
        title = doc.metadata.get("title", None)
        if not title:
            title = Path(file_path).stem
        page_count = len(doc)

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "page_number": page_num + 1,
                "content": text
            })

        pages_info.update({
            "title": title,
            "page_count": page_count,
            "pages": pages,
        })

        return pages_info
    
    def chunk_text(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks with reasonable overlap.
        """

        chunks = []
        
        if not text.strip():
            return chunks
        
        min_chunk_size = max(50, self.chunk_size // 10)
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
        
            if end < len(text):
                sentence_endings = [". ", "! ", "? "]
                sentence_positions = [text.rfind(p, start, end) for p in sentence_endings]
                sentence_positions = [pos for pos in sentence_positions if pos > start]
                
                if sentence_positions:
                    end = max(sentence_positions) + 2
                else:
                    paragraph_breaks = ["\n\n", "\r\n\r\n"]
                    paragraph_positions = [text.rfind(p, start, end) for p in paragraph_breaks]
                    paragraph_positions = [pos for pos in paragraph_positions if pos > start]
                    
                    if paragraph_positions:
                        end = max(paragraph_positions) + 2


            chunk_text = text[start:end].strip()

            if chunk_text and len(chunk_text) >= min_chunk_size:
                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "content": chunk_text
                })    
                chunk_index += 1

            overlap = min(self.chunk_overlap, self.chunk_size // 4)
            next_start = end - overlap
            if next_start <= start:
                next_start = start + min_chunk_size
            
            start = next_start
            
            if start >= len(text):
                break

        return chunks
            
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document: parse metadata and extract chunked content.
        """

        pages_info = self.extract_pages(file_path)
        title = pages_info.get('title')
        page_count = pages_info.get('page_count')
        pages = pages_info.get('pages')

        all_chunks = []
        for page in pages:
            page_chunks = self.chunk_text(page["content"], page["page_number"])
            all_chunks.extend(page_chunks)

        # print("all_chunks ",all_chunks)
        print("chunk lenght: ", len(all_chunks))

        return {
            "title": title,
            "filename": os.path.basename(file_path),
            "path": file_path,
            "page_count": page_count,
            "chunks": all_chunks
        }
