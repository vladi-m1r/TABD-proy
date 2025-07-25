from langchain_chroma import Chroma
from engine.get_embedding_function import get_embedding_function
from llama_loader import load_llama_model
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Cargar variables de entorno desde .env

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY", "")
)

def get_elementos_por_categorias(lista_categorias, chroma_path="chroma/revit"):
    """
    Devuelve un diccionario con la jerarquía:
    {categoria: [metadatas de elementos de esa categoria]}
    usando la metadata almacenada en ChromaDB.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    docs = db.get()
    metadatas = docs.get("metadatas", [])
    resultados = {}

    for categoria in lista_categorias:
        resultados[categoria] = []

    for metadata in metadatas:
        categoria = metadata.get("Categoría", "")
        if categoria in lista_categorias:
            resultados[categoria].append(metadata)

    return resultados


def get_normas_por_categoria(categoria, context="", k=6, chroma_path="chroma/normas"):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    file_metadata = {"mencion_metros": True}
    # lower case
    categoria = categoria.lower()

    if categoria == "habitaciones":
        categoria = "ambientes"

    if context != "":
        query = categoria + " " + context
    else:
        query = categoria

    results = db.similarity_search(
        query=query,
        k=k,
        filter=file_metadata
    )
    normas = []
    for doc in results:
        print("Texto:", doc.page_content)
        print(f"Source: {doc.metadata.get('source', '')}")
        print(f"Capítulo: {doc.metadata.get('capitulo', '')}")
        print(f"Artículo: {doc.metadata.get('articulo', '')}")
        print("------")
    
    contexto_unido = "\n".join([doc.page_content for doc in results])
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Contexto: " + contexto_unido + f"\n\nInstrucción: A partir del contexto, extrae y lista todas las normas que mencionen metros en la categoria: ${categoria}. Ordénalas de forma lógica, como si fueran parte de una guía de revisión para una inspección técnica en una obra. Devuelve la lista en formato numerado con una breve descripción de cada norma. Se puntual y extrae las mas claras"
            }
        ],
        #model="llama3-8b-8192",
        model="llama3-70b-8192",
    )
    normas = chat_completion.choices[0].message.content.strip().split("\n")
    return normas

def get_normas_por_categoria_RAG(categoria, context="", k=8, chroma_path="chroma/normas"):
    """
    Devuelve un diccionario con las normas de una categoría específica.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    filter_metadata = {"mencion_metros": True}
    query = context

    results = db.similarity_search(
        query=query,
        k=k,
        filter=filter_metadata
    )

    normas = []

    for doc in results:
        print("Texto:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("------")

    contexto_unido = "\n".join([doc.page_content for doc in results])

    # Estructura de prompt tipo instruct
    prompt = f"""[INST] CONTEXTO:
    {contexto_unido}
    Instrucción: A partir del contexto, extrae y lista todas las normas que mencionen metros.
    Todo en idioma español
    Ordénalas de forma lógica, como si fueran parte de una guía de revisión para una inspección técnica en una obra.
    Devuelve la lista en formato numerado con una breve descripción de cada norma. [/INST]
    """

    # Ejecutar modelo
    llm = load_llama_model()
    respuesta = llm(prompt, max_tokens=500, temperature=1.0)
    print(respuesta["choices"][0]["text"].strip())

    return normas

def get_elementos_por_categoria(categoria, chroma_path="chroma/revit"):
    """
    Devuelve una lista de elementos para una categoría específica.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    docs = db.get()
    metadatas = docs.get("metadatas", [])
    categoria = categoria.lower()
    resultados = [m for m in metadatas if m.get("Categoría", "").lower() == categoria]
    return resultados

def verificar_normas_por_categoria(categoria, normas):

    resultados = get_elementos_por_categoria(categoria=categoria)
    print(f"Resultados encontrados para la categoría '{categoria}': {len(resultados)}")
    for idx, elemento in enumerate(resultados, 1):
        print(f"Elemento {idx} metadatos: {elemento}")

    resultados_finales = []
    normas_texto = "\n".join(normas)
    for idx, elemento in enumerate(resultados, 1):
        elemento_str = str(elemento)
        prompt = f"""
        Analiza el siguiente objeto de la categoría '{categoria}':
        {elemento_str}

        Verifica el cumplimiento de las siguientes normas:
        {normas_texto}

        Se critico y con sentido común.
        Si alguna norma no aplica, indícalo explícitamente.
        Para cada norma, responde únicamente con una de las siguientes opciones: 'Cumple', 'No cumple', 'No aplica'.
        Explica brevemente el motivo de tu respuesta para cada norma.

        Devuelve la respuesta en el siguiente formato estructurado:
        Elemento: <Nombre del objeto analizado>
        Cumple:
          1. <Norma> - <Explicación breve>
          ...
        No cumple:
          1. <Norma> - <Explicación breve>
          ...
        No aplica:
          1. <Norma> - <Explicación breve>
          ...
        Si alguna sección no tiene normas, indícalo explícitamente con "Sin normas".
        """
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            #model="llama3-8b-8192",
            model="llama3-70b-8192",
        )
        resultado = chat_completion.choices[0].message.content.strip()
        print(f"Elemento {idx}\nResultado:\n{resultado}\n---")
        resultados_finales.append(resultado)
    return resultados_finales
