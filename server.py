# import os
# from flask import Flask, jsonify, request
# from http import HTTPStatus
# from icecream import ic
# import time
# import nlpdriver

# app = Flask(__name__)

# # this is for non embeddings attributes
# @app.route("/text/attributes", methods=["POST"])
# def extract_attributes():
#     return _run_nlpdriver(request.json, [nlpdriver.SUMMARY, nlpdriver.SENTIMENT, nlpdriver.KEYWORDS])

# # this is for embeddings ONLY
# @app.route("/text/embeddings", methods=["POST"])
# def extract_embeddings():
#     return _run_nlpdriver(request.json, nlpdriver.EMBEDDINGS)

# # this is for any specified attribute like embeddings, summary, sentiment, keywords
# @app.route("/text/<attribute>", methods=["POST"])
# def extract_one_attribute(attribute:str):
#     if attribute in nlpdriver.CAPABILITIES:
#         return _run_nlpdriver(request.json, [attribute])
#     else:
#         "", HTTPStatus.NOT_IMPLEMENTED

# def _run_nlpdriver(body, attrs: str|list[str]):
#     try:
#         start_time = time.time()
#         if attrs == nlpdriver.EMBEDDINGS:
#             res = nlpdriver.create_embeddings(body)
#         else:
#             res = nlpdriver.get_attributes(body, attrs)            
#         duration = time.time() - start_time
#         ic(duration, duration/len(res))
#         return jsonify(res), HTTPStatus.OK
#     except Exception as err:
#         return f"{err}", HTTPStatus.INTERNAL_SERVER_ERROR

from fastapi import FastAPI
from pydantic import BaseModel
import embeddings
from typing import Optional

class InputData(BaseModel):
    inputs: str|list[str]
    embeddings_type: Optional[str] = None


app = FastAPI(
    title="Parrotbox by Cafecit.io",
    description="This is a service to extract some of the common tasks for Project Cafecit.io such as embeddings generation, summary and topic extraction, key concepts extraction, knowledge graph generation",
    version="v0.1.0"
)

@app.post("/embedding")
@app.get("/embedding")
def create_embeddings(data: InputData):
    return embeddings.create_embeddings(data.inputs, data.embeddings_type)