from simple_ollama_rag import SimpleOllamaRag

so_rag = SimpleOllamaRag(
    inference_model="phi",
    embeddings_model="nomic-embed-text",
    tokenizer_semantic_chunk="bert-base-uncased",
    persist_directory="db",
    rag_data_directory="rag_data",
    max_tokens_embeddings=100,
    inference_config={"stop": ["\n"]},
)
so_rag.load_vectorstore()

# Ask questions
question = 'What are are not true salmon?'
response = so_rag.rag_chain(question)
print(response["message"]["content"])