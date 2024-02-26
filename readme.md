simple_ollama_rag is a simple interface for using Ollama with LangChain's RAGChain.

## Installation
Using pip:
```bash
pip install simple_ollama_rag
```
Manual:

```bash
git clone https://github.com/linkage001/simple_ollama_rag.git
cd simple_ollama_rag
pip install -r requirements.txt
```

## Usage (db on my local had a copy of the salmon article from Wikipedia)

Create the db and rag_data directories. Put your text files in the rag_data folder.

Example folder structure:
```
.
├── db
├── rag_data
│   ├── aquaculture_of_salmonids.txt
│   ├── environmental_issues_with_salmon.txt
│   ├── salmon_run.txt
│   └── salmon.txt
```
Start the ollama service with something like:

```ollama run phi```

Run:
```python
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
question = 'What are not true salmon?'
response = so_rag.rag_chain(question)
print(response["message"]["content"])

```
Output:
```output
 There are several species of fish that are colloquially called "salmon" but are not true salmon. The Danube salmon, huchen is a large freshwater salmonid closely related (from the same subfamily) to the seven species of salmon above, but others are fishes of unrelated orders, given the common name "salmon" simply due to similar shapes, behaviors and niches occupied.
```
You can create different databases and rag folders to create different experts.

If you already have data on the persist directory it will not be overwritten, you need to delete it or change de directory to create a new vector store.

## License

MIT for all code. The rag_data contains wikipedia data, so the wikipedia licence for all files inside that folder.
