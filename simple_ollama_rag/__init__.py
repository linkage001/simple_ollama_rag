from langchain_community.embeddings import OllamaEmbeddings
from .inference_tools import rag_chain
from .vectordb_tools import load_vectorstore
from .tools import decorator_timer
from .tools import decorator_timer


class SimpleOllamaRag:
    def __init__(self, inference_model, embeddings_model, tokenizer_semantic_chunk, rag_data_directory,
                 persist_directory, max_tokens_embeddings=100, inference_config=None, create_hashtags=False):
        if inference_config is None:
            inference_config = {}
        self.vectorstore = None
        self.embeddings = None
        self.retriever = None
        self.inference_model = inference_model
        self.embeddings_model = embeddings_model
        self.tokenizer_semantic_chunk = tokenizer_semantic_chunk
        self.rag_data_directory = rag_data_directory
        self.persist_directory = persist_directory
        self.max_tokens_embeddings = max_tokens_embeddings
        self.inference_config = inference_config
        self.create_hashtags = create_hashtags


    @decorator_timer
    def rag_chain(self, question, system=None, silent=True):
        retriever = self.retriever
        inference_model = self.inference_model
        inference_config = self.inference_config
        create_hashtags = self.create_hashtags
        (formatted_prompt, result), exe_time =  rag_chain(question, retriever, inference_model, inference_config, system, create_hashtags)
        if not silent:
            print(formatted_prompt)
            print(f'Time spent: {exe_time} seconds')
        return result

    def load_vectorstore(self, silent=True):
        tokenizer_semantic_chunk = self.tokenizer_semantic_chunk
        path_rag_data = self.rag_data_directory
        max_tokens_embeddings = self.max_tokens_embeddings
        persist_directory = self.persist_directory

        # Load embeddings
        embeddings = OllamaEmbeddings(model=self.embeddings_model)

        self.embeddings = embeddings

        # Load the vectorstore
        vectorstore = load_vectorstore(embeddings, self.tokenizer_semantic_chunk, self.rag_data_directory,
                                       self.max_tokens_embeddings, self.persist_directory, silent)

        if not silent:
            print('Loaded vector store')
        self.vectorstore = vectorstore

        # Create the retriever
        retriever = vectorstore.as_retriever()
        self.retriever = retriever

        return load_vectorstore(embeddings, tokenizer_semantic_chunk, path_rag_data, max_tokens_embeddings,
                                persist_directory)
