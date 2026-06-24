from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        # Check minimum article length
        if len(news.split()) < 30:
            prediction = "⚠ Please enter a complete news article (at least 30 words)."

        else:
            transformed = vectorizer.transform([news])

            result = model.predict(transformed)[0]

            confidence = model.predict_proba(transformed).max() * 100

            if result == 1:
                prediction = f"🟢 Real News ({confidence:.2f}%)"
            else:
                prediction = f"🔴 Fake News ({confidence:.2f}%)"

    return render_template(
        "index.html",
        prediction=prediction,
    news=news if request.method == "POST" else ""
)
    


if __name__ == "__main__":
    app.run(debug=True)