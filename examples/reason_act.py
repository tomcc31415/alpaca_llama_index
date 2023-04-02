import subprocess
import re

def run_llama(input_text):
    print("Running LLaMa model with input text:")
    print(input_text)
    cmd = ["./main", "-m", "./../models/gpt4all-lora-unfiltered-quantized-converted.bin", "--color", "-i", "--interactive-first", "--top_k", "10000", "--temp", "0.2", "--repeat_penalty", "1", "-t", "7", "-c", "2048", "-r", "Question:", "-r", "Observation:", "--in-prefix", " ", "-n", "-1"]
    result = subprocess.run(cmd, input=input_text.encode(), capture_output=True)
    return result.stdout.decode()

def run_python_calculation(expression):
    cmd = ["podman", "run", "--rm", "-i", "python:3.9", "python", "-c", f"import sys, math; print(eval(sys.argv[1]))", expression]
    result = subprocess.run(cmd, capture_output=True)
    return result.stdout.decode().strip()

def process_text(input_text):
    lines = input_text.split("\n")
    result_lines = []
    for line in lines:
        m = re.match(r'^Action: calculate\[(.+)\]', line)
        if m:
            expression = m.group(1)
            result = run_python_calculation(expression)
            result_lines.append(f"Observation: {result}")
        else:
            result_lines.append(line)
    return "\n".join(result_lines)

if __name__ == "__main__":
    input_text = '''Question: What is 4 * 7 / 3?
Thought: Do I need to use an action? Yes, I use calculate to do math
Action: calculate[4 * 7 / 3]
Observation:
Thought: Do I need to use an action? No, have the result
Answer: The calculate tool says it is 9.3333333333
Question: What is capital of france?
Thought: Do I need to use an action? No, I know the answer
Answer: Paris is the capital of France
Question: What is the product of pi and e
Thought: Do I need to use an action? Yes, I use calculate to do math
Action: calculate[math.pi * math.e]
Observation:'''

    processed_text = process_text(input_text)
    response = run_llama(processed_text)
    print("\nLLaMa model output:")
    print(response)

