import sys
from langchain_chroma import Chroma
from engine.get_embedding_function import get_embedding_function
from server.populate_database import populateDataBase

CHROMA_PATH = "chroma/normas"

def show_all_documents():
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    docs = db.get()
    metadatas = docs.get("metadatas", [])
    print(f"\nMostrando todos los documentos en ChromaDB ({len(metadatas)} encontrados):\n")
    for i, metadata in enumerate(metadatas, 1):
        print(f"--- Documento {i} ---")
        print(f"Source: {metadata.get('source', '')}")
        print(f"Capítulo: {metadata.get('capitulo', '')}")
        print(f"Artículo: {metadata.get('articulo', '')}")
        print(f"Texto:\n{metadata.get('text', '')}\n")
        print(f"Menciona metros: {metadata.get('mencion_metros', False)}")
        print("="*60)


def test_semantic_search(query, k=6):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query, k=k, filter={"mencion_metros": True})  # Ensure k is defined
    #results = db.max_marginal_relevance_search(query, k=k, fetch_k=k*2, filter={"mencion_metros": True})
    print(f"Resultados para la consulta: '{query}'\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Resultado {i} (score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', '')}")
        print(f"Capítulo: {doc.metadata.get('capitulo', '')}")
        print(f"Artículo: {doc.metadata.get('articulo', '')}")
        print(f"Texto:\n{doc.page_content}\n")
        print(f"Menciona metros: {doc.metadata.get('mencion_metros', False)}")
        print("="*60)
    """
    for i, doc in enumerate(results, 1):
        print(f"--- Resultado {i}) ---")
        print(f"Source: {doc.metadata.get('source', '')}")
        print(f"Capítulo: {doc.metadata.get('capitulo', '')}")
        print(f"Artículo: {doc.metadata.get('articulo', '')}")
        print(f"Texto:\n{doc.page_content}\n")
        print(f"Menciona metros: {doc.metadata.get('mencion_metros', False)}")
        print("="*60)
    """

if __name__ == "__main__":
    populateDataBase()
    print("\nEscribe tu consulta para la búsqueda semántica (escribe 'salir' para terminar):")
    while True:
        query = input("Consulta: ").strip()
        if query.lower() == "salir":
            print("Saliendo...")
            break
        if query.strip() == "-all":
            show_all_documents()
        elif query:
            test_semantic_search(query)
