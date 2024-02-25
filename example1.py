from simple_ollama_rag import SimpleOllamaRag
from simple_ollama_rag.tools import calculate_tokens_per_second
from config import (inference_config, persist_directory, rag_data_directory, max_tokens_embeddings, inference_model,
                    tokenizer_semantic_chunk, embeddings_model)

# settings
inference_config={"stop": ["\n"]}
persist_directory = "db"
rag_data_directory = "rag_data"
max_tokens_embeddings = 100

# models
inference_model = "phi"
tokenizer_semantic_chunk = "bert-base-uncased"
embeddings_model = "nomic-embed-text"


print("Inference Config:", inference_config)
print("Persist Directory:", persist_directory)
print("Path RAG Data:", rag_data_directory)
print("Max Tokens Embeddings:", max_tokens_embeddings)
print("Inference Model:", inference_model)
print("Tokenizer for Semantic Chunk:", tokenizer_semantic_chunk)
print("Embeddings Model:", embeddings_model)


def main():
    so_rag = SimpleOllamaRag(
        inference_model=inference_model,
        embeddings_model=embeddings_model,
        tokenizer_semantic_chunk=tokenizer_semantic_chunk,
        persist_directory=persist_directory,
        rag_data_directory=rag_data_directory,
        max_tokens_embeddings=max_tokens_embeddings,
        inference_config=inference_config,
    )
    so_rag.load_vectorstore(silent=False)

    # Ask questions
    questions = [
        'What happened to the Fraser River salmon population?',
        'What are the commercially important species of salmon?',
        'What are are not true salmon?',
        'How much farmed salmonoids as a percentage of the world\'s production does Norway produce?',
        'How much wild-caught fish are needed to produce 1 kg of salmon?',
        'What happens if a salmon enters a gillnet, but manages to escape?',
        'What happened to commercial salmon fisheries in California?',
        'What happens when salmon that do not find their natal rivers?',
        'Tell me about the 1982 video game called Salmon Run.',
    ]

    for question in questions:
        result = so_rag.rag_chain(question)
        tokens_per_second_load, tokens_per_second = calculate_tokens_per_second(result)

        print('=====================')
        print(f'Question: {question}')
        print(f'Answer: {result["message"]["content"]}')
        print(f'Load: {tokens_per_second_load} t/s')
        print(f'Eval: {tokens_per_second} t/s')


if __name__ == '__main__':
    main()
