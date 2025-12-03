from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import requests
import logging
from waitress import serve

# For Tesseract OCR and Image Processing
from PIL import Image
import pytesseract
import pdf2image

# For data processing terminal
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO

# For Econometrics
import statsmodels.formula.api as smf
import statsmodels.api as sm

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///admi.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Hugging Face API Key Configuration ---
app.config['HF_API_KEY'] = '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'
HF_API_URL = "https://router.huggingface.co" # CORRECTED URL

# --- Poppler and Tesseract Path Configuration ---
pdf2image.poppler_path = r"C:\Program Files\Poppler\poppler-25.11.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
logging.basicConfig(level=logging.INFO)

# --- Database Models ---
class Survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    questions = db.relationship('Question', backref='survey', lazy=True, order_by='Question.position')
    responses = db.relationship('SurveyResponse', backref='survey', lazy=True)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), nullable=False)
    options = db.Column(db.Text, nullable=True)
    validation = db.Column(db.Text, nullable=True)
    branching_logic = db.Column(db.Text, nullable=True)
    position = db.Column(db.Integer, nullable=False)
    def get_options_list(self): return json.loads(self.options) if self.options else []
    def get_validation_dict(self): return json.loads(self.validation) if self.validation else {}
    def get_branching_logic_dict(self): return json.loads(self.branching_logic) if self.branching_logic else {}

class SurveyResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    answers = db.relationship('Answer', backref='survey_response', lazy=True)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    response_id = db.Column(db.Integer, db.ForeignKey('survey_response.id'), nullable=False)
    question_id = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Text, nullable=True)

# --- Main Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/about')
def about_page(): return render_template('about.html')

@app.route('/terms')
def terms_page(): return render_template('terms.html')

@app.route('/contact')
def contact_page(): return render_template('contact.html')

@app.route('/research_management')
def research_management():
    return render_template('research_management.html')

@app.route('/surveys')
def survey_list():
    surveys = Survey.query.order_by(Survey.id).all()
    return render_template('survey_management.html', surveys=surveys)

@app.route('/surveys/new', methods=['GET', 'POST'])
def create_survey():
    if request.method == 'POST':
        new_survey = Survey(title=request.form['title'], description=request.form['description'])
        db.session.add(new_survey)
        db.session.commit()
        return redirect(url_for('edit_survey', survey_id=new_survey.id))
    return render_template('create_survey.html')

@app.route('/surveys/<int:survey_id>')
def edit_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    return render_template('edit_survey.html', survey=survey)

@app.route('/take_survey/<int:survey_id>')
def take_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    questions_data = [{'id': q.id, 'text': q.text, 'type': q.type, 'options': q.get_options_list(), 'validation': q.get_validation_dict(), 'branching_logic': q.get_branching_logic_dict()} for q in survey.questions]
    return render_template('take_survey.html', survey=survey, questions_data=questions_data)

@app.route('/document_processing')
def document_processing_page():
    return render_template('document_processing.html')

@app.route('/data_processing_terminal')
def data_processing_terminal_page():
    return render_template('data_processing_terminal.html')

# --- API Routes ---
ALLOWED_EXTENSIONS = {'pdf', 'csv', 'png', 'jpg', 'jpeg', 'webp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def query_hf_model(payload, model_url):
    """Helper function to query the Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {app.config['HF_API_KEY']}"}
    response = requests.post(model_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Hugging Face API Error: {response.status_code} {response.text}")
    return response.json()

@app.route('/api/dashboard_data')
def get_dashboard_data():
    return jsonify({
        'total_research': Survey.query.count(),
        'active_research': 0, 
        'recent_activity': [], 
        'responses_summary': {} 
    })

@app.route('/api/surveys/<int:survey_id>/questions/<int:question_id>', methods=['GET'])
def get_question(survey_id, question_id):
    question = Question.query.filter_by(survey_id=survey_id, id=question_id).first_or_404()
    return jsonify({'id': question.id, 'text': question.text, 'type': question.type, 'options': question.get_options_list(), 'validation': question.get_validation_dict(), 'branching_logic': question.get_branching_logic_dict()})

@app.route('/api/surveys/<int:survey_id>/questions', methods=['POST'])
def add_question(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    data = request.get_json()
    last_question = Question.query.filter_by(survey_id=survey_id).order_by(Question.position.desc()).first()
    new_position = (last_question.position + 1) if last_question else 0
    new_question = Question(survey_id=survey_id, text=data.get('question_text'), type=data.get('question_type'), options=json.dumps(data.get('options', [])), validation=json.dumps(data.get('validation', {})), branching_logic=json.dumps(data.get('branching_logic', {})), position=new_position)
    db.session.add(new_question)
    db.session.commit()
    return jsonify({'id': new_question.id, 'text': new_question.text, 'type': new_question.type, 'options': new_question.get_options_list(), 'validation': new_question.get_validation_dict(), 'branching_logic': new_question.get_branching_logic_dict()}), 201

@app.route('/api/surveys/<int:survey_id>/questions/<int:question_id>', methods=['PUT'])
def update_question(survey_id, question_id):
    question = Question.query.filter_by(survey_id=survey_id, id=question_id).first_or_404()
    data = request.get_json()
    question.text = data.get('question_text', question.text)
    question.type = data.get('question_type', question.type)
    question.options = json.dumps(data.get('options', question.get_options_list()))
    question.validation = json.dumps(data.get('validation', question.get_validation_dict()))
    question.branching_logic = json.dumps(data.get('branching_logic', question.get_branching_logic_dict()))
    db.session.commit()
    return jsonify({'id': question.id, 'text': question.text, 'type': question.type, 'options': question.get_options_list(), 'validation': question.get_validation_dict(), 'branching_logic': question.get_branching_logic_dict()})

@app.route('/api/surveys/<int:survey_id>/questions/<int:question_id>', methods=['DELETE'])
def delete_question(survey_id, question_id):
    question = Question.query.filter_by(survey_id=survey_id, id=question_id).first_or_404()
    db.session.delete(question)
    db.session.commit()
    return jsonify({"message": "Question deleted successfully"})

@app.route('/api/surveys/<int:survey_id>/reorder_questions', methods=['POST'])
def reorder_questions(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    new_order_ids = request.get_json().get('question_ids', [])
    questions_in_survey = {q.id: q for q in survey.questions}
    for index, q_id in enumerate(new_order_ids):
        if q_id in questions_in_survey:
            questions_in_survey[q_id].position = index
    db.session.commit()
    return jsonify({"message": "Questions reordered successfully"})

@app.route('/api/surveys/<int:survey_id>/submit_response', methods=['POST'])
def submit_survey_response(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    response_data = request.get_json()
    new_survey_response = SurveyResponse(survey_id=survey.id, timestamp=datetime.fromisoformat(response_data.get('timestamp').replace('Z', '+00:00')))
    db.session.add(new_survey_response)
    db.session.flush()
    for q_id, answer_value in response_data.get('answers', {}).items():
        value = json.dumps(answer_value) if isinstance(answer_value, list) else str(answer_value)
        new_answer = Answer(response_id=new_survey_response.id, question_id=int(q_id), value=value)
        db.session.add(new_answer)
    db.session.commit()
    return jsonify({"message": "Survey response submitted successfully", "response_id": new_survey_response.id}), 201

@app.route('/api/upload_documents', methods=['POST'])
def upload_documents():
    try:
        if 'documents' not in request.files:
            return jsonify({"error": "No document part in the request"}), 400
        files = request.files.getlist('documents')
        if not files or files[0].filename == '':
            return jsonify({"error": "No selected file"}), 400
        if len(files) > 30:
            return jsonify({"error": "Maximum 30 documents allowed at a time"}), 400
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        uploaded_filenames = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(upload_folder, filename)
                file.save(save_path)
                uploaded_filenames.append(filename)
            else:
                return jsonify({"error": f"File type not allowed: {file.filename}"}), 400
        return jsonify({"message": "Documents uploaded successfully", "filenames": uploaded_filenames}), 200
    except Exception as e:
        app.logger.error(f"An error occurred during file upload: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred during file save. Check server logs for details. Error: {str(e)}"}), 500

@app.route('/api/process_document', methods=['POST'])
def process_document():
    data = request.get_json()
    filenames, prompt, output_format = data.get('filenames'), data.get('prompt'), data.get('output_format', 'csv')
    if not filenames or not prompt: return jsonify({"error": "Missing filenames or prompt"}), 400

    try:
        filename = filenames[0]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath): return jsonify({"error": f"File not found: {filename}"}), 404

        file_extension = filename.rsplit('.', 1)[1].lower()
        extracted_text = ""

        if file_extension in ['png', 'jpg', 'jpeg', 'webp']:
            app.logger.info("Processing image file with Tesseract OCR.")
            extracted_text = pytesseract.image_to_string(Image.open(filepath))
        elif file_extension == 'pdf':
            app.logger.info("Processing PDF file with Poppler and Tesseract OCR.")
            images = pdf2image.convert_from_path(filepath, first_page=1, last_page=1)
            if not images:
                return jsonify({"error": "Could not convert PDF to image. Ensure Poppler is correctly installed."}), 500
            extracted_text = pytesseract.image_to_string(images[0])
        elif file_extension == 'csv':
            app.logger.info("Processing CSV file directly.")
            with open(filepath, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        else:
            return jsonify({"error": "Unsupported file type for processing."}), 400

        if not extracted_text.strip():
            return jsonify({"error": "Failed to extract any text from the document."}), 400

        app.logger.info("Sending extracted text to Hugging Face model.")
        system_prompt = f"You are an expert data extraction assistant. Analyze the following data and extract information based on the user's request. Provide the output ONLY in {output_format} format, without any additional explanations, introductions, or markdown formatting."
        user_prompt = f"User Request: '{prompt}'.\n\nData:\n\n{extracted_text}"
        
        model_url = HF_API_URL + "/models/mistralai/Mistral-7B-Instruct-v0.2"
        payload = {
            "inputs": f"<s>[INST] {system_prompt} [/INST]</s>[INST] {user_prompt} [/INST]",
            "parameters": {"max_new_tokens": 1500, "return_full_text": False}
        }
        
        response_data = query_hf_model(payload, model_url)
        final_output = response_data[0]['generated_text'].strip()

        if final_output.startswith(f"```{output_format}"):
            final_output = final_output[len(f"```{output_format}\n"):-3].strip()
        elif final_output.startswith("```json"):
             final_output = final_output[len("```json\n"):-3].strip()
        elif final_output.startswith("```csv"):
             final_output = final_output[len("```csv\n"):-3].strip()

        return jsonify({"result": final_output, "format": output_format}), 200

    except Exception as e:
        app.logger.error(f"An error occurred in /api/process_document: {e}", exc_info=True)
        if "poppler" in str(e).lower() or "pdf2image" in str(e).lower():
            return jsonify({"error": f"PDF processing error: Poppler might not be correctly installed or its path is wrong. Details: {str(e)}"}), 500
        else:
            return jsonify({"error": f"An unexpected error occurred. Check server logs for details."}), 500

@app.route('/api/process_data_command', methods=['POST'])
def process_data_command():
    data = request.get_json()
    csv_data, command = data.get('csv_data'), data.get('command')
    if not csv_data or not command: return jsonify({"error": "CSV data and a command are required."}), 400

    try:
        tools_description = """
        You have access to the following tools:
        1. "describe": Get descriptive statistics of the dataset. No parameters needed.
        2. "head": Get the first 5 rows of the dataset. No parameters needed.
        3. "linear_regression": Perform simple linear regression. Requires parameters "x_var" and "y_var" (column names).
        4. "ols_regression": Perform Ordinary Least Squares (OLS) regression. Requires parameters "dependent_var" (string, column name) and "independent_vars" (list of strings, column names).
        5. "logistic_regression": Perform Logistic Regression. Requires parameters "dependent_var" (string, column name, must be binary 0/1) and "independent_vars" (list of strings, column names).
        6. "histogram": Plot a histogram of a column. Requires parameter "column_name".

        Based on the user's command, decide which tool to use and what its parameters are.
        User command: "{user_command}"
        
        Respond with a single valid JSON object in the format: {{"tool": "tool_name", "parameters": {{"param1": "value1", ...}}}}
        If no tool matches, respond with: {{"tool": "unknown", "parameters": {{}}}}
        """
        
        model_url = HF_API_URL + "/models/mistralai/Mistral-7B-Instruct-v0.2"
        payload = {
            "inputs": f"<s>[INST] {tools_description.format(user_command=command)} [/INST]",
            "parameters": {"max_new_tokens": 200, "return_full_text": False}
        }
        
        response_data = query_hf_model(payload, model_url)
        tool_call_str = response_data[0]['generated_text'].strip()
        
        json_start = tool_call_str.find('{')
        json_end = tool_call_str.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise Exception(f"AI model did not return a valid JSON object. Raw response: {tool_call_str}")
        tool_call = json.loads(tool_call_str[json_start:json_end])
        
        tool_name = tool_call.get("tool")
        parameters = tool_call.get("parameters", {})

        df = pd.read_csv(BytesIO(csv_data.encode('utf-8')))
        result_text, graph_base64 = "", None

        if tool_name == "describe":
            result_text = df.describe().to_html(classes='table table-striped')
        elif tool_name == "head":
            result_text = df.head().to_html(classes='table table-striped')
        elif tool_name == "histogram":
            column_name = parameters.get("column_name")
            if column_name and column_name in df.columns:
                plt.figure(figsize=(8, 6)); df[column_name].hist(); plt.title(f'Histogram of {column_name}'); plt.xlabel(column_name); plt.ylabel('Frequency'); plt.grid(True)
                buf = BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8'); plt.close()
                result_text = f"Generated histogram for '{column_name}'."
            else:
                result_text = f"Error: Column '{column_name}' not found for histogram."
        elif tool_name == "linear_regression":
            x_var, y_var = parameters.get("x_var"), parameters.get("y_var")
            if x_var and y_var and x_var in df.columns and y_var in df.columns:
                X, y = df[[x_var]].values, df[y_var].values
                valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                X, y = X[valid_indices], y[valid_indices]
                if len(X) > 1 and len(np.unique(X)) > 1:
                    model = LinearRegression().fit(X, y)
                    result_text = f"Linear Regression ({y_var} vs {x_var}):<br>  Coefficient ({x_var}): {model.coef_[0]:.4f}<br>  Intercept: {model.intercept_:.4f}<br>  R-squared: {model.score(X, y):.4f}"
                    plt.figure(figsize=(8, 6)); plt.scatter(X, y, color='blue', label='Actual Data'); plt.plot(X, model.predict(X), color='red', label='Regression Line'); plt.title(f'Linear Regression: {y_var} vs {x_var}'); plt.xlabel(x_var); plt.ylabel(y_var); plt.legend(); plt.grid(True)
                    buf = BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8'); plt.close()
                else:
                    result_text = "Error: Not enough valid data points or unique values for regression."
            else:
                result_text = f"Error: One or both columns '{x_var}', '{y_var}' not found."
        elif tool_name == "ols_regression":
            dependent_var = parameters.get("dependent_var")
            independent_vars = parameters.get("independent_vars")
            if dependent_var and independent_vars and isinstance(independent_vars, list):
                if dependent_var in df.columns and all(col in df.columns for col in independent_vars):
                    model_cols = [dependent_var] + independent_vars
                    df_model = df[model_cols].dropna()
                    if df_model.empty:
                        result_text = "Error: No valid data points for OLS regression after handling missing values."
                    else:
                        df_model = sm.add_constant(df_model)
                        formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
                        model = smf.ols(formula=formula, data=df_model)
                        results = model.fit()
                        result_text = results.summary().as_html()
                else:
                    result_text = f"Error: One or more specified columns not found for OLS regression."
            else:
                result_text = "Error: Invalid parameters for OLS regression. Requires 'dependent_var' (string) and 'independent_vars' (list of strings)."
        elif tool_name == "logistic_regression":
            dependent_var = parameters.get("dependent_var")
            independent_vars = parameters.get("independent_vars")
            if dependent_var and independent_vars and isinstance(independent_vars, list):
                if dependent_var in df.columns and all(col in df.columns for col in independent_vars):
                    model_cols = [dependent_var] + independent_vars
                    df_model = df[model_cols].dropna()
                    if df_model.empty:
                        result_text = "Error: No valid data points for Logistic Regression after handling missing values."
                    elif not pd.api.types.is_numeric_dtype(df_model[dependent_var]) or not df_model[dependent_var].isin([0, 1]).all():
                        result_text = f"Error: Dependent variable '{dependent_var}' for Logistic Regression must be binary (0 or 1)."
                    else:
                        df_model = sm.add_constant(df_model)
                        formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
                        model = smf.logit(formula=formula, data=df_model)
                        results = model.fit()
                        result_text = results.summary().as_html()
                else:
                    result_text = f"Error: One or more specified columns not found for Logistic Regression."
            else:
                result_text = "Error: Invalid parameters for Logistic Regression. Requires 'dependent_var' (string) and 'independent_vars' (list of strings)."
        else:
            result_text = "Unknown command or could not interpret parameters. Please try again. Supported commands: 'linear regression on X vs Y', 'ols regression on Y with X1 and X2', 'logistic regression on Y with X1 and X2', 'plot histogram of Z', 'describe', 'head'."

        return jsonify({"result_text": result_text, "graph_base64": graph_base64})

    except Exception as e:
        return jsonify({"error": f"An error occurred during data processing: {str(e)}"}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    serve(app, host='0.0.0.0', port=5000)
