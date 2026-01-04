# %% package
from flask import Flask

# Flask instantation
app = Flask(__name__)

# RESTful endpoint (the parts of the URL at the end)


@app.route('/')  # We put the slash which is the same as nothing
# Wherever user is requesting this endpoint, the function below will run
def home():
    return 'Hello world'


if __name__ == '__main__':
    app.run()
