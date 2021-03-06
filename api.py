from flask import Flask
from flask_cors import CORS, cross_origin
import process_info_ret as info_proc, json, re, process_text as pt

app = Flask(__name__)
cors = CORS(app)


@app.route('/ask/<question>')
# example: /ask/where caan i learn about correct body position
def bot_request(question):
    r_list = info_proc.transform_to_lda_and_query(question)
    return json.dumps(r_list)


# if outputfile_fullname is not passed, the process text is not saved to a file
@app.route('/processcourse/<contentdirpath>/<outputfile_fullname>')
# example: /processcourse/"D:\MyDev\Working\8.4\Content\Sample Content\content\pages"/"C:\Users\testuser\Documents\desicion_engine_1\tmp\processed_text_ergo.csv"
def process_text(contentdirpath, outputfile_fullname = None):
    return pt.process_text(contentdirpath, outputfile_fullname)

