import os
from llama_index import SimpleDirectoryReader, PromptHelper
from llama_index import LLMPredictor, ServiceContext

from llama_index import (
    SimpleDirectoryReader,
    PromptHelper,
    LLMPredictor,
    ServiceContext,
    GPTListIndex
)

from GPT4ALL_LLM import GPT4ALL_LLM

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def initialize_list_index():
    max_input_size = 512
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    llm_predictor = LLMPredictor(llm=GPT4ALL_LLM())
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index_file = 'index.json'
    if os.path.exists(index_file):
        index = GPTListIndex.load_from_disk(index_file, service_context=service_context)
    else:
        # Load the your data
        documents = SimpleDirectoryReader('./data').load_data()
        index = GPTListIndex.from_documents(documents, service_context=service_context)
        index.save_to_disk(index_file)
    return index


def run_query(index, query):
    response = index.query(query)
    print(response.response)

def test_index(index):
    questions = [
        "What did Gatsby do for a living?",
        "What is the significance of the green light in The Great Gatsby?",
        "Who is the narrator in The Great Gatsby?",
        "What happens to Gatsby at the end of the novel?",
    ]

    for question in questions:
        print(f"Question: {question}")
        run_query(index, question)

def main(test=True):
    index = initialize_list_index()

    if test:
        test_index(index)
    else:
        query = "List 5 characters in the novel The Great Gatsby."
        run_query(index, query)

if __name__ == "__main__":
    main(test=False)