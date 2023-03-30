# This code creates a list index from a directory of text files
# If an index already exists, it loads it from disk
# If an index does not exist, it creates one from the text files
# It returns the index

from GPT4ALL_LLM import GPT4ALL_LLM
from llama_index import GPTListIndex, LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader
import os

def initialize_list_index():
    # Setup the prompt helper to handle the max input size, num output, and chunk overlap
    max_input_size = 512
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    # Setup the LLM predictor to use your GPT4ALL_LLM model
    llm_predictor = LLMPredictor(llm=GPT4ALL_LLM())
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