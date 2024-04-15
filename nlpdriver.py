from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from itertools import chain
from collections import Counter
from icecream import ic

# classification categories
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# functionalities and their model definition
EMBEDDINGS, SUMMARY, SENTIMENT, KEYWORDS = "embeddings", "summary", "sentiment", "keywords"
MAX_CHUNK_SIZE = {
    EMBEDDINGS: 8192,
    SUMMARY: 512,
    SENTIMENT: 512,
    KEYWORDS: 1024
}
MODEL = {
    EMBEDDINGS: "jinaai/jina-embeddings-v2-small-en", # "sentence-transformers/all-MiniLM-L6-v2",
    SUMMARY: "google/flan-t5-small",
    SENTIMENT: "SamLowe/roberta-base-go_emotions", # "distilbert/distilbert-base-uncased-finetuned-sst-2-english", #"MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    KEYWORDS: "ilsilfverskiold/tech-keywords-extractor"
}
# _MAX_TEXT_LENGTH = 4096 * 4
_MIN_TEXT_LEN_FOR_SUMMARY = 400 # roughly 75 words
_MIN_KEYWORD_LEN = 3
_MAX_KEYWORD_LEN = 25
_MAX_CHUNKS=10

# driver capabilities
CAPABILITIES = [SUMMARY, SENTIMENT, KEYWORDS, EMBEDDINGS]

# commented out text for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
# classifier = pipeline("zero-shot-classification", model=MODEL[SENTIMENT], max_length=MAX_CHUNK_SIZE[SENTIMENT])

def _create_splitter(model: str, chunk_size: int):
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer = AutoTokenizer.from_pretrained(model),
        chunk_size = chunk_size
    )

def get_sentiments(texts: list[str]) -> list[str]:
    # instantiate this for the batch so that it doesn't stay and take up memory
    classifier = pipeline("text-classification", model=MODEL[SENTIMENT], max_length=MAX_CHUNK_SIZE[SENTIMENT])
    splitter = _create_splitter(MODEL[SENTIMENT], MAX_CHUNK_SIZE[SENTIMENT])
    sentiments = []

    ic(f"creating sentiments for {len(texts)} items")
    for text in texts:
        chunks = [c for c in splitter.split_text(text)][:_MAX_CHUNKS]
        sentiments.append(Counter(item['label'] for item in classifier(chunks)).most_common(1)[0][0].capitalize())
    return sentiments
    # commented out text for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
    # data_list = [{sentiment: score for sentiment, score in zip(output['labels'], output['scores'])} for output in classifier(chunks, labels, multi_label=False)]
    # max_label = pd.DataFrame(data_list, columns = labels).sum(axis = 0).idxmax()
    # return max_label

def get_keywords(texts: list[str]) -> list[list[str]]:
    keyword_extractor = pipeline("text2text-generation", model=MODEL[KEYWORDS], max_new_tokens = 20)
    splitter = _create_splitter(MODEL[KEYWORDS], MAX_CHUNK_SIZE[KEYWORDS])
    keywords_list = []

    within_size = lambda word: (len(word) >= _MIN_KEYWORD_LEN) and (len(word) <= _MAX_KEYWORD_LEN)

    ic(f"creating keywords for {len(texts)} items")
    for text in texts:
        # current_set = set()
        chunks = [chunk for chunk in splitter.split_text(text)][:_MAX_CHUNKS]
        # current_words = [gen_text['generated_text'].split(",") for gen_text in  keyword_extractor(chunks)]
        current_words = chain(*[keyword_extractor(chunk)[0]['generated_text'].split(",") for chunk in chunks])
        current_words = list({word.strip().lower() for word in current_words if within_size(word.strip()) })
        # current_set.update(chain(*current_words))
        keywords_list.append(current_words)
    return keywords_list

def create_summaries(texts: list[str]) -> list[str]:
    # instantiate this for the batch so that it doesn't stay and take up memory
    summarizer = pipeline("text2text-generation", model=MODEL[SUMMARY], max_length = 50)
    splitter = _create_splitter(MODEL[SUMMARY], MAX_CHUNK_SIZE[SUMMARY])
    summaries = []
    
    ic(f"creating summaries for {len(texts)} items")
    for text in texts:
        if len(text) <= _MIN_TEXT_LEN_FOR_SUMMARY:
            summaries.append(text)
        else:
            chunks = ["summarize: " + chunk for chunk in splitter.split_text(text)][:_MAX_CHUNKS]
            # run it one item by one item
            chunk_summaries = [summarizer(chunk)[0]['generated_text'] for chunk in chunks]
            fix_cs = lambda cs: cs[len("Summary:"):].strip() if (cs.startswith("Summary:") or cs.startswith("summary:")) else cs.strip()
            summaries.append(" ".join([fix_cs(cs) for cs in chunk_summaries]).strip())
    return summaries

def create_embeddings(texts: list[str]) -> list[list[float]]:
    # reinstantiate this for the batch so that it doesn't stay and take up memory
    embedder = SentenceTransformer(MODEL[EMBEDDINGS])
    ic(f"creating embeddings for {len(texts)} items")
    # run one by one instead of all at a time and truncate the text approximately so that it doesn't overflow    
    return [{ EMBEDDINGS: embedder.encode(text[:MAX_CHUNK_SIZE[EMBEDDINGS]*3]).tolist() } for text in texts]
    # this is currently commented out since this is using an embedding model that can take larger text size
    # chunks = [chunk for chunk in _create_splitter(MODEL[EMBEDDINGS], MAX_CHUNK_SIZE[EMBEDDINGS]).split_text(text)]
    # embeddings = embedder.encode(chunks).tolist()
    # return [{"text": chunks[i], "embeddings": embeddings[i]} for i in range(len(chunks))]
    

# def _get_attributes_for_one(text: str, attrs: list[str] = CAPABILITIES) -> dict:    
#     res = {}
#     # do not process things longer than the max_length
#     # text = text[:_MAX_TEXT_LENGTH]
#     if SUMMARY in attrs:
#         res[SUMMARY] = create_summaries(text)
#     if SENTIMENT in attrs:
#         res[SENTIMENT] = create_sentiments(text)
#     if KEYWORDS in attrs:
#         res[KEYWORDS] = _extract_keywords(text)
#     # if EMBEDDINGS in attrs:
#     #     res[EMBEDDINGS] = _get_embeddings(text)
#     return res

def get_attributes(texts: list[str], attrs: list[str] = CAPABILITIES) -> list[dict]:
    results = [{} for text in texts]
    summaries, sentiments, keywords = [], [], []
    if SUMMARY in attrs:
        summaries = create_summaries(texts)
    if SENTIMENT in attrs:
        sentiments = get_sentiments(texts)
    if KEYWORDS in attrs:
        keywords = get_keywords(texts)

    for i in range(len(texts)):
        if summaries:
            results[i][SUMMARY] = summaries[i]
        if sentiments:
            results[i][SENTIMENT] = sentiments[i]
        if keywords:
            results[i][KEYWORDS] = keywords[i]

    return results
    # return [_get_attributes_for_one(doc, attrs) for doc in docs]