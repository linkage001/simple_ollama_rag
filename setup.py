from setuptools import setup, find_packages

setup(
    name='simple_ollama_rag',
    version='0.1.2',
    description='A simple interface for using Ollama with LangChain\'s RAGChain',
    author='gbueno86',
    packages=find_packages(),
    install_requires=[
        'chromadb',
        'huggingface-hub',
        'langchain-community',
        'langchain-core',
        'ollama',
        'semantic-text-splitter',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
