# The code below is used to generate the response to the user's input
# The prompt is the input the user gave, and we add the token "### Response:" to tell the model where to start generating
# We then run the C++ binary with the input prompt, and we get the response back
# We then return the response, which is the text that the model generated

from langchain.llms.base import LLM
import subprocess
from typing import Any, List, Mapping, Optional
DEBUG = True

class GPT4ALL_LLM(LLM):
    threads: int = 4
    batch_size: int = 8
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temp: float = 0.8
    ctx_size: int = 512
    n_predict: int = 128

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

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": "./models/gpt4all-7B/gpt4all-lora-quantized.bin"}

    @property
    def _llm_type(self) -> str:
        return "custom"