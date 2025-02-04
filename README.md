# Multi-Modal RAG with LlamaIndex

This project implements a multi-modal Retrieval-Augmented Generation (RAG) system using LlamaIndex. It can process PDF documents containing both text and images, and answer questions about their content using Google's Gemini model.

## Features

- PDF processing with text and image extraction
- Multi-modal question answering using Gemini
- Support for table extraction and processing
- Efficient document chunking and embedding
- Vector-based retrieval for relevant context

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `script.py`: Main script for processing PDFs and running the RAG pipeline
- `utils.py`: Utility functions for document processing, summarization, and response synthesis
- `requirements.txt`: Project dependencies

## Usage

1. Place your PDF document in the `sample_data` directory
2. Run the script:

```bash
python script.py
```

The script will:
1. Process the PDF and extract text, images, and tables
2. Generate embeddings and create a vector index
3. Allow you to ask questions about the document content

## How it Works

1. **Document Processing**: Uses `unstructured` library to partition PDF into chunks
2. **Embedding**: Utilizes HuggingFace's BGE embeddings for text representation
3. **Multi-modal Processing**: Handles both text and images using Gemini
4. **Retrieval**: Uses vector similarity to find relevant context
5. **Response Generation**: Synthesizes answers using retrieved context and Gemini

## Example

```python
query = "What is multihead attention?"
response = synthesize(query, retriever, llm)
print(response["response"])
```

## Dependencies

- LlamaIndex
- Unstructured
- HuggingFace Embeddings
- Google Gemini
- Pillow
- LXML
