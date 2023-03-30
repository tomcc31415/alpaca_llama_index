from GPT4ALL_LLM import GPT4ALL_LLM
from llama_index import LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader, GPTSimpleVectorIndex, LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import os


def initialize_simple_vector_index():
    # Initialize the embedding model for language chains
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    # Set up the LLM predictor to use your GPT4ALL_LLM model
    llm_predictor = LLMPredictor(llm=GPT4ALL_LLM())

    # Configure the PromptHelper to handle max input size, number of outputs, and chunk overlap
    max_input_size = 512
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    # Create a service context with default settings
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
        embed_model=embed_model
    )

    # Specify the index file
    index_file = 'index_simple_vector_index.json'

    # Check if the index file exists
    if os.path.exists(index_file):
        # Load the index if it exists
        index = GPTSimpleVectorIndex.load_from_disk(
            index_file,
            service_context=service_context
        )
    else:
        # Verify the existence of the data directory
        data_dir = './data'
        if not os.path.exists(data_dir):
            raise ValueError(f'Data directory {data_dir}does not exist')
        
        # Create a reader to load data from the data directory
        reader = SimpleDirectoryReader(data_dir)
        documents = reader.load_data()

        # Build the GPTSimpleVectorIndex from the loaded documents
        index = GPTSimpleVectorIndex.from_documents(
            documents,
            service_context=service_context
        )

        # Save the index to disk
        index.save_to_disk(index_file)

    return index

