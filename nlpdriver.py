from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from itertools import chain
from collections import Counter

# classification categories
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# functionalities and their model definition
EMBEDDINGS, SUMMARY, SENTIMENT, KEYWORDS = "embeddings", "summary", "sentiment", "keywords"
MAX_CHUNK_SIZE = {
    EMBEDDINGS: 8192,
    SUMMARY: 512,
    SENTIMENT: 512,
    KEYWORDS: 512
}
MODEL = {
    EMBEDDINGS: "jinaai/jina-embeddings-v2-base-en", # "sentence-transformers/all-MiniLM-L6-v2",
    SUMMARY: "google/flan-t5-small",
    SENTIMENT: "SamLowe/roberta-base-go_emotions", # "distilbert/distilbert-base-uncased-finetuned-sst-2-english", #"MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    KEYWORDS: "ilsilfverskiold/tech-keywords-extractor"
}
_MAX_TEXT_LENGTH = 4096 * 4

# driver capabilities
CAPABILITIES = [SUMMARY, SENTIMENT, KEYWORDS, EMBEDDINGS]

# commented out text for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
# classifier = pipeline("zero-shot-classification", model=MODEL[SENTIMENT], max_length=MAX_CHUNK_SIZE[SENTIMENT])
classifier = pipeline("text-classification", model=MODEL[SENTIMENT], max_length=MAX_CHUNK_SIZE[SENTIMENT])
keyword_extractor = pipeline("text2text-generation", model=MODEL[KEYWORDS], max_new_tokens = 20)
summarizer = pipeline("text2text-generation", model=MODEL[SUMMARY], max_length = 50)
embedder = SentenceTransformer(MODEL[EMBEDDINGS])

def _create_splitter(model: str, chunk_size: int):
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer = AutoTokenizer.from_pretrained(model),
        chunk_size = chunk_size
    )

def _get_sentiment(text: str):     
    chunks = [c for c in _create_splitter(MODEL[SENTIMENT], MAX_CHUNK_SIZE[SENTIMENT]).split_text(text)]
    return Counter(item['label'] for item in classifier(chunks)).most_common(1)[0][0].capitalize()
    # commented out text for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
    # data_list = [{sentiment: score for sentiment, score in zip(output['labels'], output['scores'])} for output in classifier(chunks, labels, multi_label=False)]
    # max_label = pd.DataFrame(data_list, columns = labels).sum(axis = 0).idxmax()
    # return max_label

def _extract_keywords(text: str) -> list[str]:
    keywords = set()
    chunks = [chunk for chunk in _create_splitter(MODEL[KEYWORDS], MAX_CHUNK_SIZE[KEYWORDS]).split_text(text)]
    keyword_list = [gen_text['generated_text'].split(",") for gen_text in  keyword_extractor(chunks)]
    keywords.update(chain(*keyword_list))
    return [keyword.strip().lower() for keyword in keywords]

def _get_summary(text: str) -> str:
    chunks = ["summarize: " + chunk for chunk in _create_splitter(MODEL[SUMMARY], MAX_CHUNK_SIZE[SUMMARY]).split_text(text)]
    return " ".join([gen_text['generated_text'] for gen_text in summarizer(chunks)] )

def get_embeddings(text: list[str]) -> list[list[float]]:
    # this is currently commented out since this is using an embedding model that can take larger text size
    # chunks = [chunk for chunk in _create_splitter(MODEL[EMBEDDINGS], MAX_CHUNK_SIZE[EMBEDDINGS]).split_text(text)]
    # embeddings = embedder.encode(chunks).tolist()
    # return [{"text": chunks[i], "embeddings": embeddings[i]} for i in range(len(chunks))]
    return [{ EMBEDDINGS: emb } for emb in embedder.encode(text).tolist()]

def _get_attributes_for_one(text: str, attrs: list[str] = CAPABILITIES) -> dict:    
    res = {}
    # do not process things longer than the max_length
    text = text[:_MAX_TEXT_LENGTH]
    if SUMMARY in attrs:
        res[SUMMARY] = _get_summary(text)
    if SENTIMENT in attrs:
        res[SENTIMENT] = _get_sentiment(text)
    if KEYWORDS in attrs:
        res[KEYWORDS] = _extract_keywords(text)
    # if EMBEDDINGS in attrs:
    #     res[EMBEDDINGS] = _get_embeddings(text)
    return res

def get_attributes(docs: list[str], attrs: list[str] = CAPABILITIES) -> list[dict]:
    return [_get_attributes_for_one(doc, attrs) for doc in docs]