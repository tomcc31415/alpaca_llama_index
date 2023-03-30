from GPT4ALL_LLM import GPT4ALL_LLM
from llama_index import GPTListIndex, LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader
import os

def initialize_list_index():
    max_input_size = 512
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    llm_predictor = LLMPredictor(llm=GPT4ALL_LLM())
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index_file = 'index_list_index.json'
    if os.path.exists(index_file):
        index = GPTListIndex.load_from_disk(index_file, service_context=service_context)
    else:
        # Load the your data
        documents = SimpleDirectoryReader('./data').load_data()
        index = GPTListIndex.from_documents(documents, service_context=service_context)
        index.save_to_disk(index_file)
    return index