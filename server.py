#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys

# def Factorial(n): # return factorial
#     result = 1
#     for i in range (1,n):
#         result = result * i
#     print("factorial is ",result)
#     return result

# print(Factorial(10))

import logging
import logging.config
import argparse

from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from flask_restful import Api, Resource 

import os
import gdown

from werkzeug.utils import secure_filename

from api import process

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from PIL import Image 

# app = Flask(__name__)
app = Flask(__name__, 
    static_url_path='/image_search/static', 
    static_folder='./static')    
api = Api(app)

UPLOAD_FOLDER = './static/uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])

class Stat(Resource):
    def get(self):
        return dict(error=0,message="server start")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_image_from_url(url, filename):
    output = f"./static/uploads/{filename}"
    gdown.download(url, output, quiet=False)

@app.route('/image_search/file', methods=['GET','POST'])
def upload_file():
    if request.method.lower() == 'post':    
        if 'file' not in request.files:            
            return jsonify(dict(error=1,message="Data invaild"))
        file = request.files['file']    
        if file.filename == '':            
            return jsonify(dict(error=1,message="Data invaild"))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            result_total, result_face  = process(image_path)
            return render_template('upload.html', rootimage=image_path,lst_result_total=result_total, lst_result_face=result_face)
    return render_template('upload.html')
            
    
# @app.route('/image_search/url', methods=['GET','POST'])
# def upload_url():
#     if request.method.lower() == 'post':    
#         if 'file' not in request.files:            
#             return jsonify(dict(error=1,message="Data invaild"))
#         file = request.files['file']    
#         if file.filename == '':            
#             return jsonify(dict(error=1,message="Data invaild"))
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)            
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             r = {"test": "okay"}
#             return jsonify(dict(error=0,data=r))
#     return render_template('upload.html')

# def test_full(filename):
#     print(procssesing_image(filename))
def main():
    api.add_resource(Stat, '/')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=9050, help='port(default: 9050)')
    args = parser.parse_args()
    port = int(args.port)
    logging.info(f"Server start: {port}")
    app.debug = True
    app.run("0.0.0.0", port=port, threaded=True)

if __name__ == "__main__":
    #filename = '21.png'
    #test_full(filename)
    main()
