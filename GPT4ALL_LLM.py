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

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": "./models/gpt4all-7B/gpt4all-lora-quantized.bin"}

    @property
    def _llm_type(self) -> str:
        return "custom"