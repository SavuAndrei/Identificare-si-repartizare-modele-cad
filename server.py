from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from clasifica import *
import os


app = Flask(__name__)

@app.route("/clasifica", methods=["POST"])
def clasifica():
    # if request.method == "GET":
    #     # Serve the page
    #     return render_template("index.html")

    # POST → handle upload

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    print(request.files['file'])

    image = request.files['file']
    filename = secure_filename(image.filename)
    image.save(os.path.join('./cache', filename))
    #print(main('./cache/' + image.filename))
    result = main('./cache/' + image.filename)
    os.remove('./cache/' + image.filename)
    return result
    # return jsonify({
    #     "filename": image.filename

    # })
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
