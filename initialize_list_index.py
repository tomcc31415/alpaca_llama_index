# This code creates a list index from a directory of text files
# If an index already exists, it loads it from disk
# If an index does not exist, it creates one from the text files
# It returns the index

from GPT4ALL_LLM import GPT4ALL_LLM
from llama_index import GPTListIndex, LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader
import os
import json

def initialize_list_index():
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

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index_file = 'index_list_index.json'
    if os.path.exists(index_file):
        # Load the index if it exists
        index = GPTListIndex.load_from_disk(index_file, service_context=service_context)
    else:
        # Load your data
        documents = SimpleDirectoryReader('./data').load_data()
        # Create the index from the data
        index = GPTListIndex.from_documents(documents, service_context=service_context)
        # Save the index to disk
        index.save_to_disk(index_file)
    return index