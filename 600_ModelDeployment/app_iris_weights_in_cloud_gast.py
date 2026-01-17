# %% package
from model_class_gast import MultiClassNet
from flask import Flask, request
import torch
import json
import requests
import os
import sys

ROOT = "/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course"
MOD_DIR = os.path.join(ROOT, "600_ModelDeployment")
if MOD_DIR not in sys.path:
    sys.path.insert(0, MOD_DIR)
# %% Download model weights file
URL = 'https://storage.googleapis.com/iris_model_gast/model_iris.pt'
r = requests.get(URL)
local_file_path = '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/600_ModelDeployment/model_iris_from_gcp.pt'

# Opening/creating file
with open(local_file_path, 'wb') as f:
    f.write(r.content)
    f.close()

# %% Model instance
model = MultiClassNet(HIDDEN_FEATURES=6, NUM_CLASSES=3, NUM_FEATURES=4)
model.load_state_dict(torch.load(local_file_path))

# %% Flask instantiation
app = Flask(__name__)

# RESTful endpoint (the parts of the URL at the end)


# Only GET and POST are allow (security reasons)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return 'Please use POST method'
    if request.method == 'POST':
        # fetch data
        data = request.data.decode('utf-8')
        dict_data = json.loads(data.replace("'", "\""))
        X = torch.tensor([dict_data['data']])
        y_test_hat_softmax = model(X)
        y_test_hat = torch.max(y_test_hat_softmax, 1)
        y_test_cls = y_test_hat.indices.cpu().detach().numpy()[0]
        cls_dict = {
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        }
        return f"Your flower belongs to {cls_dict[y_test_cls]}"
# Wherever user is requesting this endpoint, the function below will run


if __name__ == '__main__':
    app.run()

# %%
