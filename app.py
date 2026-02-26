import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from ml.visualization.ela_gradcam_pipeline import process_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ELA_FOLDER = os.path.join(BASE_DIR, "static", "ela")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ELA_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

PORT = int(os.environ.get("PORT", 5000))


@app.route("/healthz")
def healthz():
    return {"status": "healthy"}, 200


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filename = secure_filename(file.filename)

            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            label, score, ela_path, heatmap_path = process_image(upload_path)

            ela_static = None
            heatmap_static = None

            if ela_path:
                ela_static = os.path.join(ELA_FOLDER, filename)
                os.replace(ela_path, ela_static)

            if heatmap_path:
                heatmap_static = os.path.join(HEATMAP_FOLDER, filename)
                os.replace(heatmap_path, heatmap_static)

            result = {
                "label": label,
                "score": round(score, 4),
                "original": f"uploads/{filename}",
                "ela": f"ela/{filename}",
                "heatmap": f"heatmaps/{filename}" if heatmap_static else None
            }

    return render_template("index.html", result=result)


@app.route("/about-project")
def about_project():
    return render_template("about-project.html")


@app.route("/about-me")
def about_me():
    return render_template("about-me.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
