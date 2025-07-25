from llama_cpp import Llama

# Ruta al modelo GGUF ya descargado
llm = Llama(model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Contexto técnico de construcción
contexto = """
Según la norma E.070 del Reglamento Nacional de Edificaciones (RNE) del Perú, 
los muros portantes de albañilería no deben superar los 2.50 metros de altura si no están confinados. 
Además, deben tener elementos de confinamiento vertical cada 3 metros como máximo.
"""

# Pregunta basada en ese contexto
pregunta = "Tengo un muro portante de 2.0 metros de alto sin columnas de confinamiento. ¿Cumple con la norma?"

# Estructura de prompt tipo instruct
prompt = f"""[INST] CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

RESPONDE si cumple la norma. Sé específico y justifica. [/INST]
"""

# Ejecutar modelo
respuesta = llm(prompt, max_tokens=500, temperature=0.3)
print(respuesta["choices"][0]["text"].strip())
