"""
Document Processing Module
Handles PDF and DOCX extraction, cleaning, and chunking
"""
import fitz as mypdf
#import pandas as pd
#import numby as npy
from docx import document as DocxDocument
import re
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter



class DocumentProcessor:
    """Extract text from pdf and document files"""
    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from the pdf files
        Args:
            file_path:"Path to the document"
        Returns:
        Extracted text as string
        """
        if file_path.lower().endswith('pdf'):
            return DocumentProcessor.extract_from_pdf(file_path)
        if file_path.lower().endswith('dcox'):
            return DocumentProcess.extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """
        Extract text from pdf using pymupdf
        """
        doc=pymupdf.open(file_path)
        text=""

        for page_num in range(len(doc)):
            page=doc[page_num]
            text += page.get_text()
            text += f"n\n--- page(page_num+1) ---n\n"

        doc.close()
        return text
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """
        extract text from word document using docx
        """
        doc=Document(file_path)
        text=""

        #Extract Paragraphs
        for paragraph in doc.paragraph:
            text += paragraph.text + "\n"

        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
        return text

    @staticmethod
    def clean_text(text: str) -> str:
        """
        clean extracted text
        - remove white spaces
        - remove blanks
        - remove special characters
        """
        # remove white spaces
        text=re.sub(r' +', ' ', text)
        # Remove multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove tabs
        text = text.replace('\t', ' ')
        return text.strip()

class Chunking_method:
    """split documents with semantic meaning"""

    def __init__(self, chunksize: int=512, chunk_overlap: int = 50):
        """initizalize chunking
        args:
            chunksize: Target the size of chunking
            chunk_overlap: number of chrcters overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunksize, 
            chunk_overlap=chunk_overlap,
            length_function = len, 
            separators=["\n\n", "\n", ". ", " ", ""]
        )


    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        split the text in to chunks with medata 
        Args:
            Text: Text to chunk
            medata: optional 
        Returns:
            List of disctionaris with data and metadata
        """
        chunks=self.text_splitter.split_text(text)
        chunk_docs=[]
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "content": chunk,
                "metadata": {
                **(metadata or {}),
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        }
        chunk_docs.append(chunk_doc)

        return chunk_docs

    


# Example usage and testing
if __name__ == "__main__":
    #Test document proecssor
    processor = DocumentProcessor()

    #Test Chunking
    sample_text="This is a test for chunking and chunksize"*100
    chunker=Chunking_method(chunksize=100, chunk_overlap=50)
    chunks = chunker.create_chunks(sample_text, metadata={"source": "test"})

    print(f"chunksize:len{chunks} chunks")
    print(f"First_chunks:{chunks[0]['content'][:100]}...")




