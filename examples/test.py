import os
import time
import psutil
import random
import subprocess
from typing import Tuple

import database
MAX_LOAD = 4.0

def get_random_values() -> Tuple[int, float, float]:
    top_k = random.randint(1, 30000)
    top_p = round(random.uniform(0.01, 1.0), 6)
    temp = round(random.uniform(0.01, 2), 6)
    return top_k, top_p, temp

def run_program(top_k: int, top_p: float, temp: float, prompt: str) -> str:
    cmd = f"./main --color --top_k {top_k} --top_p {top_p} --temp {temp} --ctx_size 1024 -m ./../models/gpt4all-lora-unfiltered-quantized-converted.bin -f prompt.txt"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        response = result.stdout.strip()
        stripped_response = response.replace(prompt, "").strip()
    except subprocess.TimeoutExpired:
        stripped_response = ""
    
    database.insert_result(conn, cursor, temp, top_k, top_p, prompt, stripped_response)
    
    conn.commit()

    return stripped_response


if __name__ == "__main__":
    conn, cursor = database.initialize_database()

    while True:
        load = psutil.getloadavg()[0]
        if load > MAX_LOAD:
            print(f"System load is {load:.2f}. Waiting for load to decrease...")
            time.sleep(60)

        prompt = open("prompt.txt", "r").read().strip()
        top_k, top_p, temp = get_random_values()
        response = run_program(top_k, top_p, temp, prompt)

        print(f"\nResults for prompt: {prompt}")
        print(f"top_k={top_k}, top_p={top_p}, temp={temp}")
        print(response)
        time.sleep(10)

