import os
from flask import Flask, jsonify, request
from http import HTTPStatus
from icecream import ic
import time
import nlpdriver

app = Flask(__name__)

# this is for non embeddings attributes
@app.route("/text/attributes", methods=["POST"])
def extract_attributes():
    return _run_nlpdriver(request.json, [nlpdriver.SUMMARY, nlpdriver.SENTIMENT, nlpdriver.KEYWORDS])

# this is for embeddings ONLY
@app.route("/text/embeddings", methods=["POST"])
def extract_embeddings():
    return _run_nlpdriver(request.json, nlpdriver.EMBEDDINGS)

# this is for any specified attribute like embeddings, summary, sentiment, keywords
@app.route("/text/<attribute>", methods=["POST"])
def extract_one_attribute(attribute:str):
    if attribute in nlpdriver.CAPABILITIES:
        return _run_nlpdriver(request.json, [attribute])
    else:
        "", HTTPStatus.NOT_IMPLEMENTED

def _run_nlpdriver(body, attrs: str|list[str]):
    try:
        start_time = time.time()
        if attrs == nlpdriver.EMBEDDINGS:
            res = nlpdriver.get_embeddings(body)
        else:
            res = nlpdriver.get_attributes(body, attrs)            
        duration = time.time() - start_time
        ic(duration, duration/len(res))
        return jsonify(res), HTTPStatus.OK
    except Exception as err:
        return f"{err}", HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run()