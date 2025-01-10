from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable Cross-Origin Requests

# Logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# In-memory databases
users = {
    "admin": "password123",
    "doctor": "securepass"
}
appointments = []
hospital_data = None

# Load and preprocess dataset
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data = data.dropna(subset=['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Medication'])

        data['Medical Condition'] = data['Medical Condition'].str.lower()
        data['Medication'] = data['Medication'].str.lower()
        data['Gender'] = data['Gender'].str.lower()
        data['Blood Type'] = data['Blood Type'].str.upper()

        return data
    except FileNotFoundError:
        raise Exception(f"File not found at {filepath}")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

# Preprocess user input
def preprocess_input(input_data):
    try:
        return {
            "Age": input_data.get("age"),
            "Gender": input_data.get("gender").lower(),
            "Blood Type": input_data.get("blood_type").upper(),
            "Medical Condition": input_data.get("medical_condition").lower(),
        }
    except Exception as e:
        raise Exception(f"Error preprocessing input: {str(e)}")

# Compute similarity
def compute_similarity(new_patient, dataset):
    try:
        combined_dataset = dataset[['Age', 'Gender', 'Blood Type', 'Medical Condition']].astype(str)
        combined_new_patient = pd.DataFrame([new_patient]).astype(str)

        vectorizer = TfidfVectorizer()
        combined_data = combined_dataset['Medical Condition'].tolist() + combined_new_patient['Medical Condition'].tolist()
        tfidf_matrix = vectorizer.fit_transform(combined_data)

        text_similarity = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
        dataset['Similarity'] = text_similarity[0]
        return dataset.sort_values(by='Similarity', ascending=False).head(5)
    except Exception as e:
        raise ValueError(f"Error in compute_similarity: {e}")

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'pcs.html')

@app.route('/login')
def login_page():
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/about')
def about_page():
    return send_from_directory(app.static_folder, 'about.html')

@app.route('/contact')
def contact_page():
    return send_from_directory(app.static_folder, 'contact.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory(app.static_folder, 'dashboard.html')

@app.route('/appointments-page')
def appointments_page():
    return send_from_directory(app.static_folder, 'appointment.html')

@app.route('/appointments', methods=['POST'])
def create_appointment():
    try:
        data = request.json
        new_appointment = {
            "id": len(appointments) + 1,
            "patient_name": data['patient_name'],
            "doctor_name": data['doctor_name'],
            "date": data['date'],
            "time": data['time']
        }
        appointments.append(new_appointment)
        return jsonify({"message": "Appointment created", "appointment": new_appointment}), 201
    except Exception as e:
        logging.error(f"Error creating appointment: {e}")
        return jsonify({"error": "Failed to create appointment"}), 500

@app.route('/appointments', methods=['GET'])
def get_appointments():
    try:
        return jsonify(appointments)
    except Exception as e:
        logging.error(f"Error fetching appointments: {e}")
        return jsonify({"error": "Failed to fetch appointments"}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        credentials = request.json
        username = credentials.get("username")
        password = credentials.get("password")

        if username in users and users[username] == password:
            return jsonify({"success": True, "user": {"username": username}})
        else:
            return jsonify({"success": False, "message": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/similarity', methods=['POST'])
def check_similarity():
    try:
        new_patient_data = preprocess_input(request.json)
        similar_cases = compute_similarity(new_patient_data, hospital_data)
        return jsonify(similar_cases.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/appointments/<int:id>', methods=['PUT'])
def update_appointment(id):
    try:
        data = request.json
        appointment = next((appt for appt in appointments if appt['id'] == id), None)
        if not appointment:
            return jsonify({"error": "Appointment not found"}), 404

        # Update the appointment
        if 'date' in data:
            appointment['date'] = data['date']
        if 'time' in data:
            appointment['time'] = data['time']

        return jsonify({"message": "Appointment updated successfully", "appointment": appointment}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to update appointment: {e}"}), 500

@app.route('/appointments/<int:id>', methods=['DELETE'])
def delete_appointment(id):
    try:
        global appointments
        appointment = next((appt for appt in appointments if appt['id'] == id), None)
        if not appointment:
            return jsonify({"error": "Appointment not found"}), 404

        appointments = [appt for appt in appointments if appt['id'] != id]
        return jsonify({"message": "Appointment deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete appointment: {e}"}), 500

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server Error: {error}")
    return jsonify({"error": "An internal error occurred"}), 500

@app.errorhandler(404)
def not_found_error(error):
    logging.error(f"Page Not Found: {error}")
    return jsonify({"error": "Page not found"}), 404

if __name__ == '__main__':
    hospital_data = load_data("healthcare_dataset.csv")
    app.run(debug=True)