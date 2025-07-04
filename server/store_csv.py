import os
import csv
from langchain_chroma import Chroma
from langchain.schema.document import Document
from engine.get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

def clean_row(row):
    # Limpia espacios, valores nulos y normaliza claves y valores, eliminando prefijos [I] y [T]
    cleaned = {}
    for k, v in row.items():
        # Elimina prefijos [I] y [T] si existen
        if isinstance(k, str):
            key = k.strip()
            if key.startswith("[I] "):
                key = key[4:]
            elif key.startswith("[T] "):
                key = key[4:]
        else:
            key = k
        value = v.strip() if isinstance(v, str) else v
        if value not in (None, '', 'N/A', 'null', 'None'):
            cleaned[key] = value
    return cleaned

def store_csv_in_chroma(csv_path, chroma_path="chroma"):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    documents = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cleaned_row = clean_row(row)
            if not cleaned_row:
                continue
            #content = str(cleaned_row)
            #content = ", ".join(f"{k}: {v}" for k, v in cleaned_row.items())
            content = (
                f"Categoría: {cleaned_row.get('Categoría', '')}, "
                f"Elemento: {cleaned_row.get('Elemento', '')}, "
                f"Tipo: {cleaned_row.get('Tipo', '')}, "
                f"Nivel: {cleaned_row.get('Nivel', '')}, "
                f"Altura: {cleaned_row.get('Altura desconectada', '')}, "
                f"Volumen: {cleaned_row.get('Volumen', '')}, "
                f"Área: {cleaned_row.get('Área', '')}"
            )   
            doc = Document(page_content=content, metadata=cleaned_row)
            documents.append(doc)
    if documents:
        db.add_documents(documents)

def rag_query_csv(query, chroma_path="chroma", k=5):
    print("Realizando consulta RAG en ChromaDB...")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query, k=k)
    #print(results)
    context = "\n".join([doc.page_content for doc, _ in results])
    print(context)
    #prompt_template = ChatPromptTemplate.from_template(
    #    """
    #    Responde la siguiente consulta usando solo la información del contexto. Sé claro y presenta los datos en formato de tabla si es posible.\n\nContexto:\n{context}\n\n---\n\nPregunta: {question}\nRespuesta:
    #    """
    #)
    prompt_template = ChatPromptTemplate.from_template(
        """
        A partir del siguiente contexto, responde SOLO con el valor exacto del campo solicitado en la pregunta.
        Si hay varios valores, responde con una lista, uno por línea. 
        No agregues explicaciones ni texto adicional.

        Contexto:
        {context}

        ---
        Pregunta: {question}
        Respuesta:
        """
    )
    prompt = prompt_template.format(context=context, question=query)
    hf_pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=512)
    model = HuggingFacePipeline(pipeline=hf_pipe)


    respuesta = model.invoke(prompt)
    print("Respuesta RAG:\n", respuesta)
    return respuesta