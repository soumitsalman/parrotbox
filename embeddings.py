import os
from llama_cpp import Llama
import tiktoken
from functools import reduce
from retry import retry
import logging
from icecream import ic

MODEL = os.getenv("EMBEDDER_MODEL")
CONTEXT_WINDOW = int(os.getenv("EMBEDDER_CTX"))

llm = Llama(model_path=MODEL, embedding=True, verbose=False, n_ctx=CONTEXT_WINDOW)
encoding = tiktoken.get_encoding("cl100k_base")

def create_embeddings(texts: str|list[str], task_type: str = None) -> list|None:   
    if not texts:
        return None
    if isinstance(texts, str):
        texts = [texts]
    return _create_embeddings([f"{task_type}: {t}" if task_type else t for t in texts])

@retry(tries=5, logger=logging.getLogger("embeddings"))
def _create_embeddings(texts: list[str]):
    if count_tokens(texts) > CONTEXT_WINDOW:
        half = len(texts)//2
        try:
            return create_embeddings(texts[:half]) + create_embeddings(texts[half:])
        except:
            return [None]*len(texts) # insert duds
    else:
        result = [e["embedding"] for e in llm.create_embedding(texts)['data']]
        # this is a error check since sometimes the models act in benign ways and will return duds
        if len(result) != len(texts):
            raise Exception(f"Expected {len(texts)}, but generated {len(result)} embeddings.")
        return result
        
def count_tokens(texts: list[str]) -> int:
    return reduce(lambda a,b: a+b, [len(enc) for enc in encoding.encode_batch(texts)])

