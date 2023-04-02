import os
import time
import json
from initialize_faiss_index import initialize_faiss_vector_index

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def initialize_index():
    return initialize_faiss_vector_index()

def run_query(index, query):
    response = index.query(query)
    print(response.response)

def time_query(index, query):
    start_time = time.time()
    run_query(index, query)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def load_questions(file_path):
    with open(file_path, 'r') as f:
        questions = json.load(f)
    return questions

def ask_questions(index, questions):
    for q in questions:
        print(f"Question: {q['question']}")
        print(f"Expected Answer: {q['expected_answer']}")
        time_query(index, q['question'])
        print("\n")

def main():
    index = initialize_faiss_vector_index()

    questions = load_questions('questions.json')

    print("Generic Book Questions:")
    ask_questions(index, questions['generic_book_questions'])

if __name__ == "__main__":
    main()
