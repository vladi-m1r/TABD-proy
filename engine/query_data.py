import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

CHROMA_PATH = "chroma"

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

    # model = Ollama(model="mistral")
    hf_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512)
    model = HuggingFacePipeline(pipeline=hf_pipe)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()