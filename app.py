from flask import Flask, render_template, request, redirect, send_file
from functions import *
import zipfile

app = Flask(__name__)

# Define a route for the main HTML interface
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf_file = request.files["pdf_file"]
        api_key = request.form["api_key"]
        que = request.form["que"]

        # Read the PDF content from the FileStorage object
        pdf_content = pdf_file.read()
        # Your existing script to generate NLU files ba sed on the PDF
        generate_nlu_files_from_pdf(pdf_content, api_key,que )

        return redirect("/download")

    return render_template("index.html",visibility="visible")

# Define a route to download generated files
@app.route("/download")
def download():
    # Return the generated files as downloadable attachments
    nlu_files = ["nlu.yml", "stories.yml", "domain.yml"]
    zip_path = "generated_files.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in nlu_files:
            zipf.write(file)

    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
