"""
PDF Chunking Pipeline using LlamaIndex

This script implements the **PDF chunking stage** of a Retrieval-Augmented
Generation (RAG) pipeline. Its responsibility is to read PDF documents from
disk, split them into semantically meaningful chunks ("nodes"), and persist
those nodes to disk in a structured, reusable format for downstream embedding
and ingestion.

1. Recursively loads all PDF files from a configured input directory.
2. Converts each PDF into LlamaIndex `Document` objects.
3. Applies a configurable chunking strategy to split documents into nodes.
4. Filters out unwanted sections matching in DROP_TITLES.
5. Preserves metadata such as:
   - source PDF path
   - chunk index
   - chunking strategy used
6. Saves all nodes for each PDF into a single `.pkl` file.
7. Mirrors the original PDF directory structure in the output directory.

------------------------------
SUPPORTED CHUNKING STRATEGIES
------------------------------
The chunking strategy is controlled via the `CHUNKING_STRATEGY` environment
variable and supports:

- "fixed"     : Token-based fixed-size chunking
- "sentence"  : Sentence-aware chunking
- "semantic"  : Embedding-based semantic splitting (LlamaIndex)
- "recursive" : Recursive character splitting via LangChain

Only **one strategy is used per run** to allow clean comparison across
experiments.

------------------------------------------------------------
INPUT / OUTPUT STRUCTURE
------------------------------------------------------------
The input directory contains PDF documents organized in a nested folder
hierarchy. This script processes PDFs recursively, regardless of depth.

The output directory is organized by chunking strategy. For each run, all
generated chunks are written under a subdirectory named after the active
chunking strategy.

Within each strategy directory, the original input folder hierarchy is
preserved. For each PDF file, a single output file is created containing
all chunked nodes generated from that document.

Each output file:
- Corresponds to exactly one source PDF
- Contains all nodes produced for that PDF
- Is stored under the folder of the chunking strategy used
- Preserves the relative directory structure of the input PDFs

------------------------------------------------------------
CONFIGURATION
------------------------------------------------------------
All dynamic values are controlled via environment variables:

- PDF_DATA_DIR              : Root directory of input PDFs
- PDF_PARSED_OUTPUT_DIR     : Root directory for output node files. Note: for each chunking strategy, a subdirectory will be created.
- CHUNK_SIZE                : Chunk size for applicable strategies
- CHUNK_OVERLAP             : Overlap between adjacent chunks
- CHUNKING_STRATEGY         : Chunking strategy to use

This file is intended to be run as a standalone preprocessing step
before embedding and ingestion.
"""

import os
from pathlib import Path
from typing import List
import pickle

from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SemanticSplitterNodeParser,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from src.catalog.drop_titles import DROP_TITLES

load_dotenv()

#-----------------------get env variable values----------------
INPUT_DATA_DIR = os.getenv("PDF_DATA_DIR", "data/eva-docs")
OUTPUT_DATA_DIR = os.getenv("PDF_PARSED_OUTPUT_DIR", "./data/pdf_node_chunks")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

#----------------------utility functions-----------------------

def load_pdfs(data_dir: str) -> List[Document]:
    """Recursively load PDFs from data directory."""
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        required_exts=[".pdf"]
    )
    documents = reader.load_data()
    for doc in documents:
        # Get the real file path from metadata
        real_path = doc.metadata.get("file_path")
        if real_path:
            doc.metadata["source_file"] = real_path
        else:
            # fallback only if file_path isn't there
            doc.metadata["source_file"] = doc.doc_id 

    print(f"Loaded {len(documents)} PDFs from {data_dir}")
    return documents

def should_drop_titles(text: str) -> bool:
    """Checks whether a document chunk should be dropped based on title keywords."""
    lowered = text.lower()
    return any(title in lowered for title in DROP_TITLES)

#-------------------chunking strategies-----------------------
def get_node_parser(strategy: str):
    """Returns a LlamaIndex node parser based on strategy name."""
    strategy = strategy.lower()
    match strategy:
        case "fixed":
            return TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        case "sentence":
            return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        case "semantic": # this one will be expensive as it uses embedding model
            embed_model = OpenAIEmbedding(model=EMBED_MODEL, dimensions=EMBED_DIM)
            return SemanticSplitterNodeParser(
                embed_model=embed_model,
                chunk_size=CHUNK_SIZE
            )
        case "recursive":
            splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            )
            return LangchainNodeParser(splitter)
        case _:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        

#-------------------chunking pipeline--------------------
def chunk_documents_incremental():
    """Process PDFs one by one, skip already processed PDFs."""
    documents = load_pdfs(INPUT_DATA_DIR)
    parser = get_node_parser(CHUNKING_STRATEGY)
    
    total_nodes = 0
    src_root = Path(INPUT_DATA_DIR).resolve()
    output_root = Path(OUTPUT_DATA_DIR).resolve() / CHUNKING_STRATEGY
    
    for doc in documents:
        source_path = Path(doc.metadata.get("source_file")).resolve()
        rel_path = source_path.relative_to(src_root)
        out_dir = output_root / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{rel_path.stem}.pkl"

        # Skip if already processed
        if out_file.exists():
            print(f"Skipping {rel_path.name}, already processed")
            continue

        # Process nodes
        doc_nodes = parser.get_nodes_from_documents([doc])
        saved_nodes = []

        for index, node in enumerate(doc_nodes):
            text = node.get_content()
            if should_drop_titles(text):
                continue

            node.metadata.update(
                {
                    "chunking_strategy": CHUNKING_STRATEGY,
                    "chunk_index": index,
                    "source_file": doc.metadata.get("source_file", "")
                }
            )
            saved_nodes.append(node)

        # Save nodes immediately
        with open(out_file, "wb") as f:
            pickle.dump(saved_nodes, f)

        print(f"Saved {len(saved_nodes)} nodes for {rel_path.name} -> {out_file}")
        total_nodes += len(saved_nodes)

    print(f"Total chunks created in this run: {total_nodes}")
    return total_nodes

if __name__ == "__main__":
    nodes = chunk_documents_incremental()
    print(f"Total chunks created: {len(nodes)}")
    


