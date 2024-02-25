import ollama
from .tools import format_docs


def ollama_chat(question, context, inference_model, inference_config):
    formatted_prompt = f'### Context: {context}\n\n### Question: {question}\n\n'
    response = ollama.chat(model=inference_model, messages=[{'role': 'user', 'content': formatted_prompt}],
                           options=inference_config)
    return response


def rag_chain(question, retriever, inference_model, inference_config):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_chat(question, formatted_context, inference_model, inference_config)
