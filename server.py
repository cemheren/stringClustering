from gevent import monkey
monkey.patch_all()
from flask import Flask
from gevent import wsgi
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
    result = process_strings(names, False, 1, 0, 1000, 1, -1, False, True, -1)
    return json.dumps(result)

@app.route("/tags/<strings>")
def tags(strings):
    names = strings.split(" ")
    result = process_strings(names, False, 0.9, 0, 1000, 1, -1, True, False, 30)
    return json.dumps(result)

# server = wsgi.WSGIServer(('127.0.0.1', 5000), app)
server = wsgi.WSGIServer(('0.0.0.0', 5000), app)
server.serve_forever()