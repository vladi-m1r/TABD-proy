from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from server.store_csv import store_csv_in_chroma
from server.chroma_utils import get_elementos_por_categorias, get_normas_por_categoria, verificar_normas_por_categoria
from contextlib import asynccontextmanager
from server.populate_database import populateDataBase
from typing import List

class ElementosRequest(BaseModel):
    categoria: str
    contexto: str

class NormasRequest(BaseModel):
    categoria: str
    normas: List[str]
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    #print("-> Iniciando llamacpp")
    #load_llama_model()
    #print("✅ Modelo LlamaCpp/Mistral cargado.")
    print("→ Iniciando carga de PDFs...")
    populateDataBase()  # Carga los PDFs al iniciar la app
    print("→ Carga de PDFs completa.")
    yield  # Aquí arranca la app
    print("→ App cerrándose...")  # Aquí se ejecutaría al terminar
    # Puedes hacer limpieza si deseas

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/json")
async def receive_json(data: dict):
    print('JSON recibido:', data)
    return {"mensaje": "JSON recibido correctamente", "recibido": data}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(content={"error": "Nombre de archivo vacío"}, status_code=400)
    save_path = os.path.join(os.getcwd(), file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    try:
        store_csv_in_chroma(save_path)
        return {"mensaje": "CSV recibido y almacenado en ChromaDB", "archivo": file.filename}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/elementos")
async def elementos(request: ElementosRequest):
    print("JSON recibido:", request.dict())
    lista = [request.categoria] if request.categoria else []
    resultados = get_elementos_por_categorias(lista)
    # get context
    context = request.contexto if request.contexto else ""
    get_normas_por_categoria(request.categoria, context)
 
    return {"elementos": resultados}

@app.post("/actualizarNormas")
async def elementos(request: ElementosRequest):
    print("json recibido:", request.dict())

    categoria = request.categoria if request.categoria else ""
    contexto = request.contexto if request.contexto else ""
    resultados = get_normas_por_categoria(categoria=categoria, context=contexto)

    return {"elementos": resultados}

@app.post("/verificarNormas")
async def elementos(request: NormasRequest):
    print("json recibido:", request.dict())

    categoria = request.categoria if request.categoria else ""
    normas = request.normas if request.normas else []

    resultados = verificar_normas_por_categoria(categoria=categoria, normas=normas)

    return {"elementos": resultados}