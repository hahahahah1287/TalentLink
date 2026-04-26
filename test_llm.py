from llama_cpp import Llama
try:
    llm = Llama(model_path="./Qwen3.5-9B-Q5_K_M.gguf", n_gpu_layers=-1)
    print("Success")
except Exception as e:
    print(e)
