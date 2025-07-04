import argparse
from server.store_csv import rag_query_csv

CHROMA_PATH = "chroma"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Texto de consulta")
    args = parser.parse_args()
    query_text = args.query_text
    print(f"Realizando consulta RAG con el texto: {query_text}")
    rag_query_csv(query_text, chroma_path=CHROMA_PATH, k=1)

if __name__ == "__main__":
    main()
