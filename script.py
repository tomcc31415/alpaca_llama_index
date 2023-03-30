import os


from initialize_list_index import initialize_list_index

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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