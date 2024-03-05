import os
import re

from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
from langchain_community.vectorstores import Chroma
from .tools import is_directory_empty, decorator_timer

def create_chunks(tokenizer_semantic_chunk, path_rag_data, max_tokens_embeddings, persist_directory, chunk_method, separator='\n\n\n', silent=True):

    # Make a new DB if the directory is empty, else use old one
    if is_directory_empty(persist_directory):
        if not silent:
            print(f'Creating vector store with data from {path_rag_data}...')
            print(f'Chunk method: {chunk_method}')
            if chunk_method == 'separator':
                print(f'Separator: "{separator}"')

        # Make chunks
        tokenizer = Tokenizer.from_pretrained(tokenizer_semantic_chunk)
        splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)
        chunks = ['']

        # Read the files
        for filename in os.listdir(path_rag_data):
            path = os.path.join(path_rag_data, filename)
            with open(path, mode='r', encoding='utf=8') as f:
                text = f.read()

            # Is separated by separator
            if chunk_method == 'separator':
                chunks = [*chunks, *text.split(separator)]



            # Is semantic
            else:
                @decorator_timer
                def text_split(_text, split_max_chat_count):
                    return [_text[i: i + split_max_chat_count] for i in range(0, len(text), split_max_chat_count)]

                splits, exe_time = text_split(text, 10000)
                total_splits = len(splits)
                if not silent:
                    print(f'Created {total_splits} splits from {path}')
                    print(f'Time spent: {exe_time} seconds')
                for i, split in enumerate(splits):
                    current_chunks = len(chunks)
                    chunks = [*chunks, *splitter.chunks(split, max_tokens_embeddings)]
                    if not silent:
                        print(
                            f'Created chunks {len(chunks) - current_chunks} for split {i}/{total_splits} of {path}. Total chunks: {len(chunks)}')
        if not silent:
            print(f'Total chunks: {len(chunks)}')

        return chunks

    else:
        return []

def load_vectorstore(embeddings, persist_directory, chunks,
                     silent=True):
    if is_directory_empty(persist_directory):
    # Create Ollama embeddings and vector store
        if not silent:
            print(f'Creating vector store...')
        @decorator_timer
        def create_vector_store():
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_directory)
            vectorstore.persist()
            return vectorstore

        vectorstore, exe_time = create_vector_store()
        if not silent:
            print(f'Vector store created and saved to {persist_directory}')
            print(f'Time spent: {exe_time} seconds')
    else:
        if not silent:
            print('Loading vector store...')

        @decorator_timer
        def load_vector_store():
            vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
            return vectorstore

        vectorstore, exe_time = load_vector_store()
        if not silent:
            print(f'Time spent: {exe_time} seconds')

    return vectorstore
