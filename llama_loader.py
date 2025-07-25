from llama_cpp import Llama

llm = None

def load_llama_model():
    global llm
    if llm is None:
        llm = Llama(
            model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=10,
            verbose=False
        )
    return llm