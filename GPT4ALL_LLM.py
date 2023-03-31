# The code below is used to generate the response to the user's input
# The prompt is the input the user gave, and we add the token "### Response:" to tell the model where to start generating
# We then run the C++ binary with the input prompt, and we get the response back
# We then return the response, which is the text that the model generated

from langchain.llms.base import LLM
import subprocess
from typing import Any, List, Mapping, Optional
DEBUG = False

class GPT4ALL_LLM(LLM):
    threads: int = 4
    batch_size: int = 8
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temp: float = 0.8
    ctx_size: int = 512
    n_predict: int = 128

    def __init__(self, threads: int = None, batch_size: int = None, repeat_last_n: int = None,
                 repeat_penalty: float = None, temp: float = None, ctx_size: int = None, n_predict: int = None):
        super().__init__()  # call the parent class's constructor
        if threads is not None:
            self.threads = threads
        if batch_size is not None:
            self.batch_size = batch_size
        if repeat_last_n is not None:
            self.repeat_last_n = repeat_last_n
        if repeat_penalty is not None:
            self.repeat_penalty = repeat_penalty
        if temp is not None:
            self.temp = temp
        if ctx_size is not None:
            self.ctx_size = ctx_size
        if n_predict is not None:
            self.n_predict = n_predict

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        # Add the prompt to the beginning of the generated text
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

        try:
            # Run the C++ binary and capture the output
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        except Exception as e:
            print("Error running the C++ binary:", e)
            return ""

        # Remove the prompt from the output
        response = result.stdout
        response = response[len(prompt):]

        # Log input and output if DEBUG flag is set
        if DEBUG:
            print("Input prompt:", prompt)
            print("Command output:", result.stdout)
            print("Cleaned response:", response)

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": "./models/gpt4all-7B/gpt4all-lora-quantized.bin"}

    @property
    def _llm_type(self) -> str:
        return "GPT4ALL_LLM"
