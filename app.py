from flask import Flask, jsonify, request, make_response, send_file
from predict_new import *
import time
import os
from dotenv import load_dotenv
from flask_httpauth import HTTPBasicAuth
import pandas as pd
import logging

app = Flask(__name__)
auth = HTTPBasicAuth()
load_dotenv(dotenv_path='.env', verbose=True)

logging.basicConfig(level=logging.DEBUG)

usernames_env = os.getenv("user_names").split(';')
pwds_env = os.getenv("passwords").split(';')

@auth.verify_password
def verify_password(username, password):
    if username in usernames_env:
        idx = usernames_env.index(username)
        if password == pwds_env[idx]:
            return True
    return False

# Option 1: Generate Manual Test Cases in Excel
@app.route('/generate_test_cases', methods=['POST'])
@auth.login_required
def generate_test_cases():
    try:
        args = request.json
        logging.debug(f'Test Cases API Input Received: {args}')
        
        st = time.time()
        test_cases = generate_test_cases_json(args['user_story'], args['acceptance_criteria'], args['temperature'])
        et = time.time()
        logging.debug(f'Time Taken: {et - st}')
        
        # Convert the test cases into a DataFrame and export to Excel
        df = pd.DataFrame(test_cases['test_cases'])
        output_file = "/tmp/test_cases.xlsx"
        df.to_excel(output_file, index=False)
        
        # Return the generated file as a download
        return send_file(output_file, as_attachment=True, download_name="manual_test_cases.xlsx")
    
    except Exception as e:
        logging.error(f"Error in generating test cases: {str(e)}")
        return jsonify({'error': 'Failed to generate test cases'}), 500

# Option 2: Generate Java Code and BDD Feature (Upload Excel)
@app.route('/generate_java_bdd', methods=['POST'])
@auth.login_required
def generate_java_bdd():
    try:
        file = request.files['test_cases_file']
        
        if not file.filename.endswith('.xlsx'):
            return jsonify({'error': 'Invalid file type, please upload an Excel file.'}), 400
        
        logging.debug(f'Excel File Received: {file.filename}')
        
        st = time.time()
        
        file_path = "/tmp/test_cases.xlsx"
        file.save(file_path)
        
        feature, glue = get_feature_glue_from_excel(file_path)
        
        feature_file = "/tmp/feature_file.feature"
        glue_file = "/tmp/glue_file.java"
        
        with open(feature_file, 'w') as f:
            f.write(feature)
        
        with open(glue_file, 'w') as f:
            f.write(glue)
        
        et = time.time()
        logging.debug(f'Time Taken: {et - st}')
        
        # Provide download links for both files
        return jsonify({
            'feature_download_link': '/download_file?file=feature_file',
            'glue_download_link': '/download_file?file=glue_file'
        })
    
    except Exception as e:
        logging.error(f"Error in generating Java and BDD files: {str(e)}")
        return jsonify({'error': 'Failed to process the uploaded Excel file'}), 500

# Route to download files
@app.route('/download_file', methods=['GET'])
@auth.login_required
def download_file():
    try:
        file_name = request.args.get('file')
        file_map = {
            'feature_file': '/tmp/feature_file.feature',
            'glue_file': '/tmp/glue_file.java'
        }
        
        file_path = file_map.get(file_name)
        if file_path and os.path.exists(file_path):
            response = send_file(file_path, as_attachment=True)
            os.remove(file_path)  
            return response
        else:
            return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        logging.error(f"Error in downloading file: {str(e)}")
        return jsonify({'error': 'Failed to download file'}), 500

@app.route('/health', methods=['GET'])
@auth.login_required
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
