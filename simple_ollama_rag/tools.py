import os
from time import time

def decorator_timer(some_function):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time()-t1
        return result, end
    return wrapper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def is_directory_empty(directory):
    if os.path.isdir(directory):
        if not os.listdir(directory):
            return True
        else:
            return False
    else:
        return False


def calculate_tokens_per_second(json_data):
    load_duration = json_data['load_duration']
    eval_duration = json_data['eval_duration']
    prompt_eval_count = json_data['prompt_eval_count']
    eval_count = json_data['eval_count']

    tokens_per_second_load = prompt_eval_count / (load_duration / 1000000000)
    tokens_per_second = eval_count / (eval_duration / 1000000000)

    return tokens_per_second_load, tokens_per_second
