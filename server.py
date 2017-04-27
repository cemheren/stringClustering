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
    result = process_strings(names, False, 1, 0, 1000, 1, -1, False, True, False, 5)
    return json.dumps(result)

@app.route("/tags/<strings>")
def tags(strings):
    names = strings.split(" ")
    result = process_strings(names, False, 0.9, 0, 1000, 1, -1, True, False, True, 30)
    return json.dumps(result)


if __name__ == "__main__":
    # app.run() # local
    app.run(host='0.0.0.0') # cloud