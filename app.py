import pandas as pd
from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
import joblib
import os
import traceback
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Add a secret key for flash messages

# Load models and symptom vocab once on startup
try:
    model = joblib.load('./models/model.pkl')
    label_encoder = joblib.load('./models/label_encoder.pkl')
    symptom_list = joblib.load('./models/symptom_vocab.pkl')
except Exception as e:
    print(f"Error loading models: {str(e)}")
    model = None
    label_encoder = None
    symptom_list = None

@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, HTTPException):
        return e
    # Log the error
    print(f"Error: {str(e)}")
    print(traceback.format_exc())
    return render_template('error.html', error="An unexpected error occurred. Please try again later."), 500

def validate_models():
    if model is None or label_encoder is None or symptom_list is None:
        raise Exception("Required models are not loaded properly")

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        validate_models()
        prediction = None
        if request.method == 'POST':
            symptoms = [request.form.get(f'symptom{i}') for i in range(1, 18)]
            symptoms = [s.lower().strip() for s in symptoms if s and s.strip() != '']
            
            if not symptoms:
                flash('Please select at least one symptom', 'error')
                return render_template('index.html', symptom_list=symptom_list)

            input_vector = [0] * len(symptom_list)
            for symptom in symptoms:
                if symptom in symptom_list:
                    idx = symptom_list.index(symptom)
                    input_vector[idx] = 1

            pred_label = model.predict([input_vector])[0]
            prediction = label_encoder.inverse_transform([pred_label])[0]

        return render_template('index.html', symptom_list=symptom_list, prediction=prediction)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return render_template('error.html', error="An error occurred while processing your request. Please try again."), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    try:
        validate_models()
        prediction = None
        if request.method == 'POST':
            symptoms = request.form.getlist('symptoms')
            symptoms = [s.lower().strip() for s in symptoms if s and s.strip() != '']
            
            if not symptoms:
                flash('Please select at least one symptom', 'error')
                return render_template('predict.html', symptoms_list=symptom_list)

            input_vector = [0] * len(symptom_list)
            for symptom in symptoms:
                if symptom in symptom_list:
                    idx = symptom_list.index(symptom)
                    input_vector[idx] = 1

            pred_label = model.predict([input_vector])[0]
            disease = label_encoder.inverse_transform([pred_label])[0]
            
            prediction = {
                'disease': disease,
                'confidence': 90,  # dummy value
                'symptoms': symptoms
            }

        return render_template('predict.html', symptoms_list=symptom_list, prediction=prediction)
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return render_template('error.html', error="An error occurred while processing your request. Please try again."), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)

app.run(debug=True)




# import pandas as pd
# from flask import Flask, render_template, request
# import joblib

# app = Flask(__name__)

# # Load the trained model, label encoder, and symptom vocabulary
# model = joblib.load('./models/model.pkl')
# label_encoder = joblib.load('./models/label_encoder.pkl')
# symptom_list = joblib.load('./models/symptom_vocab.pkl')  # Assuming you saved symptom list here

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction = None
#     if request.method == 'POST':
#         # Get selected symptoms from the form
#         symptoms = [request.form.get(f'symptom{i}') for i in range(1, 18)]
#         # Clean: lowercase, strip, and remove empty
#         symptoms = [s.lower().strip() for s in symptoms if s and s.strip() != '']

#         # Create a binary input vector matching symptom_list order and length
#         input_vector = [0] * len(symptom_list)
#         for symptom in symptoms:
#             if symptom in symptom_list:
#                 idx = symptom_list.index(symptom)
#                 input_vector[idx] = 1
#             else:
#                 # You may want to handle unknown symptoms here
#                 pass

#         # Predict encoded label
#         pred_label = model.predict([input_vector])[0]

#         # Decode label to disease name
#         prediction = label_encoder.inverse_transform([pred_label])[0]

#     # Render the template with symptom_list and prediction if any
#     return render_template('index.html', symptom_list=symptom_list, prediction=prediction)

# # @app.route('/about')
# # def about():
#     # return render_template('about.html')

# # @app.route('/contact')
# # def contact():
#     # return render_template('contact.html')
# if __name__ == '__main__':
#     app.run(debug=True)