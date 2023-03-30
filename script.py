import os
import subprocess
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from llama_index import SimpleDirectoryReader, LLMPredictor, PromptHelper, GPTSimpleVectorIndex
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set the debug flag
DEBUG = False

class LLaMALLM(LLM):
    threads: int = 4
    batch_size: int = 8
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temp: float = 0.8
    ctx_size: int = 512
    n_predict: int = 128

    def _call(self, prompt, stop=None):
        prompt += "### Response:"

        # Prepare the command to run the C++ binary
        command = [
            "./main",
            "-m",
            "./models/gpt4all-7B/gpt4all-lora-quantized.bin",
            "-p",
            prompt,
            "-n",
            str(self.n_predict),
            "--threads",
            str(self.threads),
            "--batch_size",
            str(self.batch_size),
            "--repeat_last_n",
            str(self.repeat_last_n),
            "--repeat_penalty",
            str(self.repeat_penalty),
            "--temp",
            str(self.temp),
            "--ctx_size",
            str(self.ctx_size),
        ]

        if DEBUG:
            print("Input Prompt:", prompt)

        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        except Exception as e:
            print("Error running the C++ binary:", e)
            return ""

        response = result.stdout
        response = response[len(prompt):]

        if DEBUG:
            print("Model Response:", response)

        return response

    def _llm_type(self):
        return "custom"


def initialize_index():
    max_input_size = 1024
    num_output = 512
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    documents = SimpleDirectoryReader('data').load_data()
    llm_predictor = LLMPredictor(llm=LLaMALLM())

    index_file = 'index.json'

    if os.path.exists(index_file):
        index = GPTSimpleVectorIndex.load_from_disk(index_file, embed_model=embed_model, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    else:
        index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)
        index.save_to_disk(index_file)
        index = GPTSimpleVectorIndex.load_from_disk(index_file, embed_model=embed_model, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    return index

def run_query(index, query):
    response = index.query(query)
    print(response.response)

def test_index(index):
    questions = [
        "Describe Gatsby in one sentence.",
        "What did Gatsby do for a living?",
        "What is the significance of the green light in The Great Gatsby?",
        "Who is the narrator in The Great Gatsby?",
        "What happens to Gatsby at the end of the novel?",
    ]

    for question in questions:
        print(f"Question: {question}")
        run_query(index, question)


def main(test=False):
    index = initialize_index()

    if test:
        test_index(index)
    else:
        query = "Describe Gatsby in one sentence."
        run_query(index, query)

if __name__ == "__main__":
    main(test=True)