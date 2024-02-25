import os
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
from langchain_community.vectorstores import Chroma
from .tools import is_directory_empty


def load_vectorstore(embeddings, tokenizer_semantic_chunk, path_rag_data, max_tokens_embeddings, persist_directory,
                     silent=True):
    # Make a new DB if the directory is empty, else use old one
    if is_directory_empty('db'):
        if not silent:
            print('Creating vector store...')
        # Make chunks
        tokenizer = Tokenizer.from_pretrained(tokenizer_semantic_chunk)
        splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)

        # Read the files
        chunks = ['']
        for filename in os.listdir(path_rag_data):
            path = os.path.join(path_rag_data, filename)
            with open(path, mode='r', encoding='utf=8') as f:
                text = f.read()
                chunks = [*chunks, *splitter.chunks(text, max_tokens_embeddings)]

        # # Create Ollama embeddings and vector store
        vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_directory)
        vectorstore.persist()
    else:
        if not silent:
            print('Loading vector store...')
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    return vectorstore
