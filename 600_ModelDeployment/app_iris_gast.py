# %% package
from model_class_gast import MultiClassNet
from flask import Flask
import torch

import os
import sys

ROOT = "/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course"
MOD_DIR = os.path.join(ROOT, "600_ModelDeployment")
if MOD_DIR not in sys.path:
    sys.path.insert(0, MOD_DIR)


# %% Model instance

# Flask instanciation
app = Flask(__name__)

# RESTful endpoint (the parts of the URL at the end)


@app.route('/')  # We put the slash which is the same as nothing
# Wherever user is requesting this endpoint, the function below will run
def home():
    return 'Hello world'


if __name__ == '__main__':
    app.run()
