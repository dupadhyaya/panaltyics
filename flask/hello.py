# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:34:55 2019

@author: du
"""

from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {name}!'