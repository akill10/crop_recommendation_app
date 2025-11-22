from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib, numpy as np, json, os, hashlib
from deep_translator import GoogleTranslator

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load model
model = joblib.load("crop_model.pkl")

# Crop info for 22 crops
crop_data = {
    "rice": {"tip": "Rice grows best in clayey soil with plenty of water and high humidity.",
             "fert": "Use urea and potash-rich fertilizers regularly for higher yield."},
    "maize": {"tip": "Maize requires warm weather, moderate rainfall, and loamy soil.",
              "fert": "Apply DAP and potash; avoid overwatering the plants."},
    "chickpea": {"tip": "Chickpea thrives in dry, cool climates and sandy soil.",
                 "fert": "Use phosphate fertilizer before sowing for strong roots."},
    "kidneybeans": {"tip": "Kidney beans prefer cool weather and well-drained loamy soil.",
                    "fert": "Use compost and potash fertilizers for better yield."},
    "pigeonpeas": {"tip": "Pigeon peas need warm, semi-arid climates and loamy soil.",
                   "fert": "Use phosphate and organic manure during early stages."},
    "mothbeans": {"tip": "Moth beans grow well in dry, sandy soil with little rainfall.",
                  "fert": "Add nitrogen and phosphate fertilizers moderately."},
    "mungbean": {"tip": "Mung beans prefer warm weather and sandy loam soil.",
                 "fert": "Use DAP and compost before sowing."},
    "blackgram": {"tip": "Black gram requires warm, humid weather and loamy soil.",
                  "fert": "Apply nitrogen and phosphate fertilizers early."},
    "lentil": {"tip": "Lentil grows best in cool, dry climate and sandy soil.",
               "fert": "Add phosphate fertilizer before sowing for better pods."},
    "pomegranate": {"tip": "Pomegranate grows in hot, dry climate and loamy soil.",
                    "fert": "Use nitrogen and potash during flowering stage."},
    "banana": {"tip": "Banana requires humid climate and rich, organic soil.",
               "fert": "Add potassium and compost regularly."},
    "mango": {"tip": "Mango grows in hot, dry climates and loamy soil.",
              "fert": "Apply nitrogen and potash before flowering season."},
    "grapes": {"tip": "Grapes grow in hot, dry climate and well-drained soil.",
               "fert": "Use potash and compost for stronger vines."},
    "watermelon": {"tip": "Watermelon needs sandy soil and warm temperatures.",
                   "fert": "Use potash and compost for sweeter fruits."},
    "muskmelon": {"tip": "Muskmelon grows well in sandy loam soil with good drainage.",
                  "fert": "Use nitrogen and compost during early growth."},
    "apple": {"tip": "Apple needs cool climate and loamy soil with good drainage.",
              "fert": "Apply organic manure and potash before flowering."},
    "orange": {"tip": "Orange prefers subtropical climate and sandy loam soil.",
               "fert": "Use nitrogen and potash fertilizers twice yearly."},
    "papaya": {"tip": "Papaya grows in warm climate and fertile, well-drained soil.",
               "fert": "Use nitrogen and phosphate for better fruiting."},
    "coconut": {"tip": "Coconut thrives in sandy coastal soil with high humidity.",
                "fert": "Use organic compost and potash twice a year."},
    "cotton": {"tip": "Cotton grows well in black soil under warm conditions.",
               "fert": "Use NPK mix fertilizers during flowering."},
    "jute": {"tip": "Jute needs humid, warm climate and loamy soil.",
             "fert": "Use organic manure and potash before sowing."},
    "coffee": {"tip": "Coffee requires shade, humid weather, and rich soil.",
               "fert": "Use nitrogen and compost regularly to maintain yield."}
}

users_file = "users.json"

# Helpers
def load_users():
    if not os.path.exists(users_file):
        with open(users_file, "w") as f:
            json.dump({}, f)
    try:
        with open(users_file, "r") as f:
            data = f.read().strip()
            return json.loads(data) if data else {}
    except json.JSONDecodeError:
        return {}

def save_users(users):
    with open(users_file, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def translate_text(text, lang):
    if lang in ["english", "en"]:
        return text
    try:
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except Exception:
        return text

# Routes
@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("predict_page"))
    return render_template("login.html")

@app.route("/signup", methods=["GET"])
def signup_page():
    return render_template("signup.html")

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form["username"].strip()
    password = request.form["password"].strip()
    users = load_users()
    if username in users:
        return jsonify({"status": "error", "message": "⚠️ User already exists!"})
    users[username] = hash_password(password)
    save_users(users)
    return jsonify({"status": "success", "message": "✅ Account created successfully!"})

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"].strip()
    password = request.form["password"].strip()
    users = load_users()
    if username not in users:
        return jsonify({"status": "error", "message": "❌ User not found!"})
    if users[username] != hash_password(password):
        return jsonify({"status": "error", "message": "❌ Incorrect password!"})
    session["user"] = username
    if "language" not in session:
        session["language"] = "english"
    return jsonify({"status": "success", "redirect": url_for("predict_page")})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/set_language", methods=["POST"])
def set_language():
    lang = request.json.get("lang", "english")
    session["language"] = lang
    return jsonify({"status": "ok"})

@app.route("/predict_page")
def predict_page():
    if "user" not in session:
        return redirect(url_for("index"))
    lang = session.get("language", "english")
    return render_template("index.html", lang=lang)

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("index"))
    vals = [float(request.form[x]) for x in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    input_data = np.array([vals])
    pred = model.predict(input_data)[0].lower()

    info = crop_data.get(pred, {"tip": "No information available.", "fert": "Use balanced NPK fertilizers."})
    lang = session.get("language", "english")

    return render_template("result.html",
                           crop=pred,
                           tip=info["tip"],
                           fertilizer=info["fert"],
                           lang=lang)

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")
    lang = data.get("lang", "english")
    return jsonify({"translated": translate_text(text, lang)})

if __name__ == "__main__":
    app.run(debug=True)
