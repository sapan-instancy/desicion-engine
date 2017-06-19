from flask import Flask
from flask_cors import CORS, cross_origin
import process_info_ret as info_proc, json, re

app = Flask(__name__)
cors = CORS(app)

@app.route('/ask/<question>')
def bot_request(question):
    r_list = info_proc.transform_to_tfidf(question)
    return json.dumps(r_list)