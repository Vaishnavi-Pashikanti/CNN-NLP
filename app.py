from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Use a Smaller Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base", torch_dtype=torch.float16
).to(device)

# ✅ Optimize Model Execution
model = torch.compile(model)

def vqa_pipeline(image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    
    answer = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files or "question" not in request.form:
            return jsonify({"error": "Missing image or question"})
        
        image = request.files["image"]
        question = request.form["question"]
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(image_path)
        
        answer = vqa_pipeline(image_path, question)
        return jsonify({"answer": answer, "image_url": image_path})
    
    return render_template("index.html")

# ✅ Remove duplicate route definitions
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
