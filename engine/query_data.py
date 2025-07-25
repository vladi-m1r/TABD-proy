import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

from llama_cpp import Llama

CHROMA_PATH = "chroma"
MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Ajusta la ruta si es necesario

PROMPT_TEMPLATE = """
Responde la siguiente pregunta usando solo la información del contexto. Sé detallado y explica tu respuesta.

Contexto:
{context}

---

Pregunta: {question}
Respuesta:
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Usar llama_cpp para cargar el modelo GGUF
    llm = Llama(model_path=MODEL_PATH)
    respuesta = llm(prompt, max_tokens=512, temperature=0.3)
    response_text = respuesta["choices"][0]["text"].strip()

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()