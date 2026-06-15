from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():

    result = ""
    confidence = ""
    explanation = ""

    if request.method == 'POST':

        news = request.form['news']

        transformed = vectorizer.transform([news])

        prediction = model.predict(transformed)[0]

        probabilities = model.predict_proba(transformed)[0]
        confidence = round(max(probabilities) * 100, 2)

        if prediction == "fake":
            result = "⚠️ Likely Fake News"
            explanation = "This article contains patterns commonly found in misinformation."
        else:
            result = "✅ Likely Real News"
            explanation = "This article contains patterns commonly found in reliable news."

    return render_template(
        'index.html',
        result=result,
        confidence=f"{confidence}%",
        explanation=explanation
    )

if __name__ == '__main__':
    app.run(debug=True)