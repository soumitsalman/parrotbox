from transformers import AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import logging



# classification categories
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# functionalities and their model definition
EMBEDDINGS, SUMMARY, SENTIMENT, KEYWORDS = "embeddings", "summary", "sentiment", "keywords"
MAX_CHUNK_SIZE = {
    EMBEDDINGS: 512,
    SUMMARY: 4096,
    SENTIMENT: 512,
    KEYWORDS: 1024
}
MODEL = {
    EMBEDDINGS: "sentence-transformers/all-MiniLM-L6-v2",
    SUMMARY: "google/flan-t5-base",
    SENTIMENT: "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    KEYWORDS: "ilsilfverskiold/tech-keywords-extractor"
}
_MAX_TEXT_LENGTH = 4096 * 4

# driver capabilities
CAPABILITIES = [SUMMARY, SENTIMENT, KEYWORDS]

classifier = pipeline("zero-shot-classification", model=MODEL[SENTIMENT], max_length=MAX_CHUNK_SIZE[SENTIMENT])
keyword_extractor = pipeline("text2text-generation", model=MODEL[KEYWORDS], max_new_tokens = 50)
summarizer = pipeline("summarization", model=MODEL[SUMMARY], max_length = 120)

def _create_splitter(model: str, chunk_size: int):
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer = AutoTokenizer.from_pretrained(model),
        chunk_size = chunk_size
    )

def _get_classification(text: str, labels: list[str] = SENTIMENT_LABELS):    
    # determine sentiment and topic  
    # classifier = pipeline("zero-shot-classification", model=MODEL[SENTIMENT], max_length=MAX_CHUNK_SIZE[SENTIMENT])
    outputs = [classifier(c, labels, multi_label=False) for c in _create_splitter(MODEL[SENTIMENT], MAX_CHUNK_SIZE[SENTIMENT]).split_text(text)]
    data_list = [{sentiment: score for sentiment, score in zip(output['labels'], output['scores'])} for output in outputs]
    max_label = pd.DataFrame(data_list, columns = labels).sum(axis = 0).idxmax()
    return max_label

def _extract_keywords(text: str) -> list[str]:
    keywords = set()
    # keyword_extractor = pipeline("text2text-generation", model=MODEL[KEYWORDS], max_new_tokens = 5)
    for chunk in _create_splitter(MODEL[KEYWORDS], MAX_CHUNK_SIZE[KEYWORDS]).split_text(text):
        keywords.update(keyword_extractor(chunk)[0]['generated_text'].split(', '))
    return [keyword.lower() for keyword in keywords]

def _get_summary(text: str) -> str:
    # summarizer = pipeline("text2text-generation", model=MODEL[SUMMARY], max_length = 100) 
    return summarizer("summarize: " + text)[0]['generated_text']

def get_attributes_for_one(text: str, attrs: list[str] = CAPABILITIES) -> dict:    
    res = {}
    # do not process things longer than the max_length
    text = text[:_MAX_TEXT_LENGTH]
    if SUMMARY in attrs:
        res[SUMMARY] = _get_summary(text)
    if SENTIMENT in attrs:
        res[SENTIMENT] = _get_classification(text, SENTIMENT_LABELS)
    if KEYWORDS in attrs:
        res[KEYWORDS] = _extract_keywords(text)
    return res

def get_attributes_for_many(docs: list[str], attrs: list[str] = CAPABILITIES) -> list[dict]:
    return [get_attributes_for_one(doc, attrs) for doc in docs]