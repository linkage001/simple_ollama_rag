from simple_ollama_rag import SimpleOllamaRag
from simple_ollama_rag.tools import calculate_tokens_per_second

# settings
inference_config={"stop": [
    # "\n",
    "<|im_end|>",
    "<|im_start|>",
    "GGGG",
    "User: ",
    "Question:",
    "Assistant:",
]}
persist_directory = "db"
rag_data_directory = "rag_data"
max_tokens_embeddings = 100

# models
inference_model = "phi"
tokenizer_semantic_chunk = "bert-base-uncased"
embeddings_model = "nomic-embed-text"


def main():
    so_rag = SimpleOllamaRag(
        inference_model=inference_model,
        embeddings_model=embeddings_model,
        tokenizer_semantic_chunk=tokenizer_semantic_chunk,
        persist_directory=persist_directory,
        rag_data_directory=rag_data_directory,
        max_tokens_embeddings=max_tokens_embeddings,
        inference_config=inference_config,
        create_hashtags=False
    )
    so_rag.load_vectorstore(silent=False)

    # Ask questions

    questions_fish = [
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

    for question in questions_fish:
        result, exe_time = so_rag.rag_chain(question, system="You are a helpful chatbot.",silent=True)
        tokens_per_second_load, tokens_per_second = calculate_tokens_per_second(result)

        print('=====================')
        print(f'Question: {question}')
        print(f'Answer: {result["message"]["content"]}')
        print(f'Load: {tokens_per_second_load} t/s')
        print(f'Eval: {tokens_per_second} t/s')
        print(f'Time spent: {exe_time} seconds')
        print('')


if __name__ == '__main__':
    main()
