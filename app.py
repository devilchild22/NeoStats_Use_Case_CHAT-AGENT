from flask import Flask, render_template, request, jsonify
import os
from app.components.agent.agent import answer_query
from app.utils.process_doc.processing import process_document
from app.utils.logger.logger import get_logger


app = Flask(__name__)

vectorstore = None

logger = get_logger()


@app.route("/")
def home():

   return render_template(
        "index.html"
    )


@app.route("/api/chat", methods=["GET", "POST"])
def chat():                              
    global vectorstore

    if request.method == "POST":  

        question = request.form.get("question")
        print(question)

        mode = request.form.get("mode")
        print(mode)

        name = request.form.get("name")

        result = answer_query(user_query = question, mode = mode, thread_id = name, vectorstore = vectorstore)

        logger.info(f"The final respinse sendign to frontend is : {result["answer"]}")

        logger.info(f"The final respinse sendign to frontend is : {result["sources"]}")

    return jsonify({
        "answer" : result["answer"],
        "sources" : result["sources"]
    })



@app.route("/api/upload", methods=["PUT"])
def document_upload():
    file = request.files.get("file")  

    if not file:
        return {"error": "No file provided"}, 400

    global vectorstore

    try:
        logger.info(f"File received: {file.filename}")
        vectorstore = process_document(file)

        if vectorstore is None:
            return {"error": "Failed to process document"}, 500

        return {"message": "Document processed successfully",
                "chunks":  vectorstore.index.ntotal}, 200

    except ValueError as e:
        return {"error": str(e)}, 415
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return {"error": "Internal server error"}, 500
    
    


if __name__ == "__main__":

    app.run(debug=False)