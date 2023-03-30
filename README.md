# GPT Language Model Index

This is a Python script that creates and tests a language model index based on the GPT architecture. It uses the Langchain library to create the index, which is stored in a JSON file. The index is based on a pre-trained GPT model called "gpt4all-lora-quantized.bin", and the script defines a custom Langchain LLM class to interact with the C++ binary that runs the model. The LLM class specifies parameters such as the batch size, temperature, and context size, which affect how the model generates responses.

## Requirements

To run this script, you will need:

- Python 3.x
- The Langchain library (`pip install langchain`)
- The Transformers library (`pip install transformers`)

## Usage

To run the script, navigate to the directory where it is located and run:

```
python gpt_index.py
```


By default, the script will run in test mode, which tests the index with a set of pre-defined questions. If you want to run a custom query instead, change the `test` argument in the `main()` function to `False`, and edit the `query` variable to contain your query string.

## License

This script is licensed under the MIT License. Feel free to modify and use it as you see fit.