from flask import Flask
from logic import *
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/cluster/<strings>")
def cluster(strings):
    names = strings.split(" ")
    result = process_strings(names)
    return json.dumps(result)

@app.route("/name/<strings>")
def name(strings):
    names = strings.split(" ")
    result = process_strings(names, False, 1, 0.01, 1000, 1, -1, False, True, False, 5)
    return json.dumps(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0')