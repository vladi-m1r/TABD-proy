import os
import shutil
import fitz  # PyMuPDF
from langchain.schema.document import Document
from engine.get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import re

CHROMA_PATH = "chroma/normas"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "server/data"

# Heurística para detectar si un texto contiene una tabla
def contiene_tabla(texto):
    lineas = texto.splitlines()
    filas_tabla = [l for l in lineas if '\t' in l or l.count('  ') > 2]
    return len(filas_tabla) >= 2

# Normaliza ligaduras y caracteres raros en el texto
def normalizar_ligaduras(texto):
    texto = texto.replace('\x02 ', 'fi')
    return texto

def populateDataBase():
    # Create (or update) the data store.
    documents = load_documents()
    # Agrupar por source (archivo PDF)
    from collections import defaultdict
    docs_by_source = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get("source", "")
        docs_by_source[source].append(doc)

    article_chunks = []
    for source, docs_list in docs_by_source.items():
        # Eliminar las dos primeras líneas de cada página y filtrar líneas basura
        def clean_page(text):
            lines = text.splitlines()
            # Quitar las primeras cuatro líneas
            lines = lines[4:] if len(lines) > 4 else []
            filtered = []
            for l in lines:
                s = l.strip()
                # Filtrar líneas basura tipo '*UiILFR$' y similares
                if re.match(r'^\*UiILFR.', s):
                    continue
                # Filtrar líneas que parecen artefactos de imágenes: muchas mayúsculas y números, o pocos espacios y muchos símbolos
                if (len(s) > 10 and sum(c.isupper() for c in s) > 5 and sum(c.isdigit() for c in s) > 2) or re.match(r'^[A-Za-z0-9\-_]{8,}$', s):
                    continue
                # Filtrar líneas con muchos caracteres no alfabéticos y pocos espacios (probable base64, hash, o nombre de imagen)
                if len(s) > 10 and sum(not c.isalnum() and not c.isspace() for c in s) > 5:
                    continue
                filtered.append(l)
            # Unir líneas que terminan en incisos (a), b), 45.1, 45.1., etc. con la siguiente línea
            joined = []
            i = 0
            while i < len(filtered):
                line = filtered[i]
                # Si la línea termina en inciso (letra o número seguido de ')', número con punto decimal, o número con punto decimal y punto final)
                if re.match(r".*((\b[a-zA-Z]\)|\b\d+\.\d+\.?))$", line.strip()) and i+1 < len(filtered):
                    # Unir con la siguiente línea
                    line = line.rstrip() + ' ' + filtered[i+1].lstrip()
                    i += 1  # Saltar la siguiente línea
                joined.append(line)
                i += 1
            return "\n".join(joined)
        cleaned_pages = [clean_page(d.page_content) for d in docs_list]
        full_text = "\n".join(cleaned_pages)
        # Usar la metadata de la primera página como base
        base_metadata = docs_list[0].metadata.copy()
        # Crear un documento "virtual" con el texto completo
        full_doc = Document(page_content=full_text, metadata=base_metadata)
        # Fragmentar por artículos y capítulos
        article_chunks.extend(split_by_article(full_doc))
    add_to_chroma(article_chunks)

def load_documents():
    # Usar PyMuPDF para extraer solo texto real de los PDFs
    docs = []
    print("Ruta absoluta a DATA_PATH:", os.path.abspath(DATA_PATH))
    print("Contiene archivos:", os.listdir(DATA_PATH))
    for fname in os.listdir(DATA_PATH):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(DATA_PATH, fname)
        pdf = fitz.open(path)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")
            # Guardar cada página como un Document
            docs.append(Document(
                page_content=text,
                metadata={"source": f"data/{fname}", "page": str(page_num+1)}
            ))
    return docs


# Fragmenta un documento por artículos y capítulos, enriqueciendo la metadata
import re
def split_by_article(document: Document):
    text = normalizar_ligaduras(document.page_content)
    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")
    # Buscar capítulos y artículos
    # Ejemplo: CAPÍTULO V ... Artículo 27.- ...
    pattern = r'(CAP[IÍ]TULO\s+[IVXLCDM]+.*?)(?=CAP[IÍ]TULO|$)'  # Capítulos
    capitulos = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    docs = []
    for cap in capitulos:
        cap_match = re.search(r'CAP[IÍ]TULO\s+[IVXLCDM]+', cap, re.IGNORECASE)
        capitulo = cap_match.group(0) if cap_match else ""
        articulos = re.split(r'(Art[íi]culo\s+\d+\.-)', cap, flags=re.IGNORECASE)
        for i in range(1, len(articulos), 2):
            encabezado = articulos[i]
            contenido = articulos[i+1] if i+1 < len(articulos) else ""
            articulo = encabezado.strip()
            texto_articulo = normalizar_ligaduras(contenido.strip())
            # Detectar número de artículo
            articulo_num_match = re.search(r'Art[íi]culo\s+(\d+)', articulo, re.IGNORECASE)
            articulo_num = articulo_num_match.group(1) if articulo_num_match else None
            if articulo_num:
                # Buscar incisos tipo '5.1', '5.2', '27.3', etc. al inicio de línea, con o sin punto final
                inciso_regex = rf'^{articulo_num}\.\d+\.?'
                inciso_matches = list(re.finditer(inciso_regex, texto_articulo, re.MULTILINE))
                if inciso_matches:
                    inciso_indices = [m.start() for m in inciso_matches]
                    inciso_indices.append(len(texto_articulo))  # Para el último inciso
                    for idx in range(len(inciso_indices)-1):
                        start = inciso_indices[idx]
                        end = inciso_indices[idx+1]
                        inciso_text = texto_articulo[start:end].strip()
                        inciso_encabezado_match = re.match(inciso_regex, inciso_text)
                        inciso_encabezado = inciso_encabezado_match.group(0) if inciso_encabezado_match else ""
                        mencion_metros = bool(re.search(r"(\d+(\.\d+)?\s?(m(etro|etros)?))|((metro|metros))", inciso_text, re.IGNORECASE))
                        tabla = contiene_tabla(inciso_text)
                        tabla_texto = ""
                        texto_final = inciso_text
                        if tabla:
                            lineas = inciso_text.splitlines()
                            filas_tabla = [l for l in lineas if '\t' in l or l.count('  ') > 2]
                            tabla_texto = '\n'.join(filas_tabla)
                            if tabla_texto and tabla_texto not in inciso_text:
                                texto_final = f"{inciso_text}\nTABLA:\n{tabla_texto}"
                        metadata = {
                            "capitulo": capitulo,
                            "articulo": articulo,
                            "inciso": inciso_encabezado,
                            "source": source,
                            "page": page,
                            "mencion_metros": mencion_metros,
                            "contiene_tabla": tabla,
                            "tabla_texto": tabla_texto,
                            "text": texto_final
                        }
                        docs.append(Document(page_content=texto_final, metadata=metadata))
                    continue  # Ya procesamos los incisos, no guardar el artículo completo
            # Si no hay incisos, guarda el artículo completo como antes
            mencion_metros = bool(re.search(r"(\d+(\.\d+)?\s?(m(etro|etros)?))|((metro|metros))", texto_articulo, re.IGNORECASE))
            tabla = contiene_tabla(texto_articulo)
            tabla_texto = ""
            texto_final = texto_articulo
            if tabla:
                lineas = texto_articulo.splitlines()
                filas_tabla = [l for l in lineas if '\t' in l or l.count('  ') > 2]
                tabla_texto = '\n'.join(filas_tabla)
                if tabla_texto and tabla_texto not in texto_articulo:
                    texto_final = f"{texto_articulo}\nTABLA:\n{tabla_texto}"
            metadata = {
                "capitulo": capitulo,
                "articulo": articulo,
                "inciso": "",
                "source": source,
                "page": page,
                "mencion_metros": mencion_metros,
                "contiene_tabla": tabla,
                "tabla_texto": tabla_texto,
                "text": texto_final
            }
            docs.append(Document(page_content=texto_final, metadata=metadata))
    return docs


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    import re
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

        # Detect if the chunk mentions meters (variants: 'numero m', 'metro', 'metros', 'm')
        chunk.metadata["mencion_metros"] = bool(re.search(r"(\d+(\.\d+)?\s?(m(etro|etros)?))|((metro|metros))", chunk.page_content, re.IGNORECASE))

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
