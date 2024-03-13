import os
from flask import Flask, jsonify, request
from http import HTTPStatus
from icecream import ic
import time
import nlpdriver

app = Flask(__name__)

@app.route("/text/attributes", methods=["POST"])
def extract_all_attributes():
    return _extract_attributes(request.json, nlpdriver.CAPABILITIES)

@app.route("/text/<attribute>", methods=["POST"])
def extract_one_attribute(attribute:str):
    if attribute in nlpdriver.CAPABILITIES:
        return _extract_attributes(request.json, [attribute])
    else:
        "", HTTPStatus.NOT_IMPLEMENTED

def _extract_attributes(body, attrs: list[str]):
    try:
        start_time = time.time()
        res = nlpdriver.get_attributes_for_many(body, attrs)
        duration = time.time() - start_time
        ic(duration, duration/len(res))
        return jsonify(res), HTTPStatus.OK
    except Exception as err:
        return f"{err}", HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run()