from GPT4ALL_LLM import GPT4ALL_LLM
from llama_index import LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader, GPTFaissIndex, LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import faiss
import json
import os

def initialize_faiss_vector_index():
    # Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Configure the PromptHelper to handle max input size, number of outputs, and chunk overlap
    prompt_helper = PromptHelper(
        config['max_input_size'],
        config['num_output'],
        config['max_chunk_overlap']
    )
    # Set up the LLM predictor to use your GPT4ALL_LLM model
    llm_predictor = LLMPredictor(llm=GPT4ALL_LLM(
        threads=config.get('threads', 4),
        batch_size=config.get('batch_size', 8),
        repeat_last_n=config.get('repeat_last_n', 64),
        repeat_penalty=config.get('repeat_penalty', 1.1),
        temp=config.get('temp', 0.8),
        ctx_size=config.get('ctx_size', 512),
        n_predict=config.get('n_predict', 128)
    ))

    # Initialize the embedding model for language chains
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    # Create a faiss index
    d = 768
    faiss_index = faiss.IndexFlatL2(d)

    # Create a service context with default settings
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
        embed_model=embed_model
    )

    # Specify the index file
    index_file = 'index_faiss_vector_index.json'

    # Check if the index file exists
    if os.path.exists(index_file):
        # Load the index if it exists
        index = GPTFaissIndex.load_from_disk(
            index_file,
            service_context=service_context,
            faiss_index=faiss_index
        )
    else:
        # Verify the existence of the data directory
        data_dir = './data'
        if not os.path.exists(data_dir):
            raise ValueError(f'Data directory {data_dir} does not exist')

        # Create a reader to load data from the data directory
        reader = SimpleDirectoryReader(data_dir)
        documents = reader.load_data()

        # Build the GPTFaissIndex from the loaded documents
        index = GPTFaissIndex.from_documents(
            documents,
            faiss_index=faiss_index,
            service_context=service_context
        )

        # Save the index to disk
        index.save_to_disk(index_file)

    return index