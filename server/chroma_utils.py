from langchain_chroma import Chroma
from engine.get_embedding_function import get_embedding_function
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Cargar variables de entorno desde .env
#model = "llama-3.3-70b-versatile"  # Modelo por defecto
model = "moonshotai/kimi-k2-instruct"  # Modelo por defecto

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
        model=model,
    )
    normas = chat_completion.choices[0].message.content.strip().split("\n")
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
        Actúa como un verificador técnico especializado en normativas de edificación. A continuación, se proporciona una lista de normas técnicas y un conjunto de parámetros de un elemento arquitectónico.

        Tu tarea es:

        1. Revisar si cada norma se cumple según los parámetros dados.
        2. Indicar si la norma se cumple o no se cumple.
        3. Justificar tu respuesta con base en la comparación directa.
        4. Si la norma no aplica a los parámetros dados, indícalo.
        5. Usa una lista enumerada con la evaluación de cada norma.

        ### Normas:
        {normas_texto}

        ### Parámetros del elemento:
        {elemento_str}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
        )
        resultado = chat_completion.choices[0].message.content.strip()
        print(f"Elemento {idx}\nResultado:\n{resultado}\n---")
        nombre_elemento = elemento.get("Elemento", f"elemento_{idx}")
        id_elemento = elemento.get("ID", f"id_{idx}")
        resultados_finales.append({
            "elemento": nombre_elemento,
            "id": id_elemento,
            "resultado": resultado
        })
    return resultados_finales
