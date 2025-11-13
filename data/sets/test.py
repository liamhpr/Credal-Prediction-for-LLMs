import json
import os 

"""
File for testing what data json.load returns
"""

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, 'coqa-dev-v1.0.json')

with open(path, 'r') as infile : 
    data = json.load(infile)['data']
    print(data[:1])
