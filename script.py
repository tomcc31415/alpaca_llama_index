# This code creates an index, and queries it using a query string. The query string is passed to the run_query function, which calls the query method on the index. The response from the query method is printed.
# The index is created using the initialize_list_index function.
# The initialize_list_index function creates an index, and adds documents to it.
# The query method of the index returns a SearchResponse object, which has a response attribute.
# The response attribute is a list of dict objects, each of which represents a document in the index.
# The query string is a natural language query.
# The query string is a list query, which is a type of query that returns a list of documents.
# The query string specifies that the list should contain 5 characters in the novel The Great Gatsby.

import os

from initialize_simple_vector_index import initialize_simple_vector_index
#from initialize_list_index import initialize_list_index

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_query(index, query):
    response = index.query(query)
    print(response.response)

def main():
    index= initialize_simple_vector_index()

    query = "List 5 characters in the novel The Great Gatsby."
    run_query(index, query)

if __name__ == "__main__":
    main()

