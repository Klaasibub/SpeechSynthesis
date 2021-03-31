from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Very small, very ez!"

if __name__ == "__main__":
    app.run(port=4567)
