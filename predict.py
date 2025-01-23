import os
import pandas as pd
import openai
from io import BytesIO
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv(dotenv_path='.env', verbose=True)

azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_model = os.getenv("AZURE_OAI_MODEL")

openai.api_type = "azure"
openai.api_base = azure_oai_endpoint
openai.api_version = "2023-12-01-preview"
openai.api_key = azure_oai_key

def allowed_file(filename):
    """Check if the uploaded file is a valid Excel file."""
    if '.' not in filename:
        return False
    file_extension = filename.rsplit('.', 1)[1].lower()
    return file_extension in {'xls', 'xlsx', 'csv'}

def read_file(filepath):
    """Read Excel or CSV files."""
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            return pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def generate_manual_test_cases(user_story, acceptance_criteria):
    """
    Use OpenAI to generate manual test cases based on the user story and acceptance criteria.
    """
    prompt = f"""
    Generate detailed test cases with the following structure:
    - **Test Case ID**: A unique identifier for each test case.
    - **Description**: A clear description of what the test case is verifying.
    - **Preconditions**: Any conditions that must be met before performing the test.
    - **Test Steps**: Steps to perform the test.
    - **Expected Results**: What should happen if the test passes.

    User Story: {user_story}
    Acceptance Criteria: {acceptance_criteria}
    """
    
    try:
        response = openai.Completion.create(
            engine=azure_oai_model,
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7
        )
        test_cases = response.choices[0].text.strip().split('\n')
        
        test_case_list = []
        for i, test_case in enumerate(test_cases, start=1):
            test_case_list.append({
                'Test Case ID': i,
                'Description': test_case,
                'Preconditions': f"Precondition for {test_case}",
                'Steps': f"Step 1: {test_case}",
                'Expected Result': f"Expected result for {test_case}"
            })
        
        return test_case_list
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return []
    except Exception as e:
        print(f"An error occurred while generating test cases: {e}")
        return []

def generate_bdd_feature_and_java(filepath):
    """
    Generate BDD feature file and Java code from the uploaded Excel test cases using OpenAI.
    """
    try:
        df = read_file(filepath)
        if df is None:
            return None, None

        required_columns = ['Test Case ID', 'Description', 'Steps', 'Expected Result']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {', '.join(missing_columns)}")
            return None, None
        
        # Example prompt for generating a BDD feature file and Java code based on the test cases
        prompt = f"Generate a BDD feature file and corresponding Java code based on the following test cases:\n\n{df.to_string(index=False)}\n\nBDD Feature and Java Code:"
        
        # Use Azure OpenAI's GPT model to generate the BDD feature and Java code
        response = openai.Completion.create(
            engine=azure_oai_model,
            prompt=prompt,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7
        )

        result = response.choices[0].text.strip()
        bdd_feature, java_code = result.split('\n\n')
        
        return bdd_feature, java_code
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred while generating BDD feature and Java code: {e}")
        return None, None

def clean_openai_response(response_text):
    """Clean up the response from OpenAI."""
    cleaned_response = response_text.strip()
    if '\n\n' in cleaned_response:
        return cleaned_response.split('\n\n')
    return cleaned_response

def generate_all_features(user_story, acceptance_criteria, file_path):
    """Generate manual test cases and BDD features concurrently."""
    with ThreadPoolExecutor() as executor:
        manual_test_cases_future = executor.submit(generate_manual_test_cases, user_story, acceptance_criteria)
        bdd_feature_future = executor.submit(generate_bdd_feature_and_java, file_path)

        manual_test_cases = manual_test_cases_future.result()
        bdd_feature, java_code = bdd_feature_future.result()

    return manual_test_cases, bdd_feature, java_code

# Main function that can be used to interact with the script
if __name__ == "__main__":
    user_story = "As a user, I want to be able to log in so that I can access my account."
    acceptance_criteria = "1. The system should accept a valid username and password. 2. The system should deny access with incorrect credentials."
    
    test_cases = generate_manual_test_cases(user_story, acceptance_criteria)
    
    if test_cases:
        df = pd.DataFrame(test_cases)
        excel_file = BytesIO()
        df.to_excel(excel_file, index=False, engine='openpyxl')
        excel_file.seek(0)
        
        with open("generated_test_cases.xlsx", "wb") as f:
            f.write(excel_file.read())
        
        print("Manual test cases generated and saved as 'generated_test_cases.xlsx'.")
    
    # Generating BDD feature and Java code:
    file_path = "test_cases.xlsx"  
    
    bdd_feature, java_code = generate_bdd_feature_and_java(file_path)
    
    if bdd_feature and java_code:
        # Save the BDD feature and Java code
        with open("bdd_feature.txt", "w") as bdd_file:
            bdd_file.write(bdd_feature)
        
        with open("java_code.java", "w") as java_file:
            java_file.write(java_code)
        
        print("BDD feature file and Java code generated and saved.")
    else:
        print("Failed to generate BDD feature and Java code.")
