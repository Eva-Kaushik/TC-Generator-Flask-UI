import os
import requests
import json
import openai
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv(dotenv_path='.env', verbose=True)
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_model = os.getenv("AZURE_OAI_MODEL")

openai.api_type = "azure"
openai.api_base = azure_oai_endpoint
openai.api_version = "2023-12-01-preview"
openai.api_key = azure_oai_key

df = pd.read_excel('TestData.xlsx')

def get_openai_response(system_msg, prompt, examples=None, temp=0):
    msgs = [{"role": "system", "content": system_msg}]
    if examples is not None:
        for k, v in examples.items():
            msgs.extend([{"role": "user", "content": k}, {"role": "assistant", "content": v}])
    msgs.append({"role": "user", "content": prompt})
    
    response = openai.ChatCompletion.create(
        engine=azure_oai_model,
        messages=msgs,
        temperature=temp,
        max_tokens=4000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    finish_reason = response['choices'][0]['finish_reason']
    text = response['choices'][0]['message']['content'].replace(' .', '.').strip()
    msgs.append({"role": "assistant", "content": text})
    
    if finish_reason == 'length':
        while True:
            msgs.append({"role": "user", "content": "continue"})
            response = openai.ChatCompletion.create(
                engine=azure_oai_model,
                messages=msgs,
                temperature=temp,
                max_tokens=4000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            text1 = response['choices'][0]['message']['content'].replace(' .', '.').strip()
            text += text1
            msgs.append({"role": "assistant", "content": text1})
            finish_reason = response['choices'][0]['finish_reason']
            if finish_reason != 'length':
                break     
    print(f'Token Usage:\n{response["usage"]}')
    return text

def read_examples(typ):
    if typ == 'glue':
        egs = dict(zip(df['feature'].values, df['glue'].values))
    elif typ == 'feature':
        egs = dict(zip(df['manual_test_cases_text'].values, df['feature'].values))
    elif typ == 'manual_test':
        egs = {}
        for _, row in df.iterrows():
            user_story = row['user_story']
            acceptance_criteria = row['acceptance_criteria']
            prompt1 = f'User Story:\n {user_story} \nAcceptance Criteria:\n {acceptance_criteria}'
            egs[prompt1] = row['manual_test_cases_json']
    else:
        egs = None
    return egs

def generate_test_cases_json(user_story, acceptance_criteria, temp=0):
    system_msg1 = '''As a quality assurance specialist, you are tasked with creating detailed step-by-step test cases for the given user story and its acceptance criteria. The output format of the test cases should be in JSON. Use 'NA' when there is no input_test_data. Here is an example of how the JSON structure should look like:
    {
        "test_cases": [
            {
                "test_case_no": 1,
                "test_case_description": "Test case description 1",
                "test_steps": [
                    {
                        "test_step_no": 1,
                        "test_step_description": "Test step description 1",
                        "input_test_data": "Input test data 1",
                        "expected_results": "Expected results 1"
                    },
                    {
                        "test_step_no": 2,
                        "test_step_description": "Test step description 2",
                        "input_test_data": "Input test data 2",
                        "expected_results": "Expected results 2"
                    }
                ]
            }
        ]
    }'''
    prompt1 = f'User Story:\n {user_story} \nAcceptance Criteria:\n {acceptance_criteria}'
    tc_human_json = get_openai_response(system_msg=system_msg1, prompt=prompt1, temp=temp)
    try:
        tc_human_json = json.loads(tc_human_json)
    except ValueError as e:
        print(f"Invalid JSON: {e}")
        tc_human_json = tc_human_json.replace('""', '"')
        try:
            tc_human_json = json.loads(tc_human_json)
        except ValueError as e:
            print(f"Invalid JSON after replacement: {e}")
            print(tc_human_json)
    return tc_human_json

def convert_json_to_text(json_data):
    text = f"Test Case {json_data['test_case_no']}:\n{json_data['test_case_description']}\nTest Steps:\n"
    for step in json_data['test_steps']:
        text += f"{step['test_step_no']}. Description: {step['test_step_description']}\n"
        text += f"   - Expected Results: {step['expected_results']}\n"
        text += f"   - Input Test Data: {step['input_test_data']}\n"
    return text

def get_feature(tc_json_list, temp=0):
    df = pd.DataFrame(tc_json_list)
    df = df.replace("null", np.nan)
    df = df[df.notnull().all(1)]
    df.reset_index(inplace=True, drop=True)
    df['test_case_no'] = range(1, df.shape[0]+1)
    tc_json_list = df.to_dict(orient='records')
    tc_text_all = ''
    for tc_json in tc_json_list:
        tc_text = convert_json_to_text(tc_json)
        tc_text_all += tc_text
    
    feature_msg2 = "As a QA specialist, create a BDD feature file for the given test cases. Include necessary scenarios, steps, and assertions to comprehensively cover the functionality. Ensure that the input test data and any website links mentioned in the test cases are explicitly used in the feature file under examples section with placeholders in the steps."
    feature_resp = get_openai_response(system_msg=feature_msg2, prompt=tc_text_all, temp=temp)
    return feature_resp

def get_glue(feature_resp, temp=0):
    glue_msg2 = "As a QA specialist, generate complete glue file for all scenarios in the given feature file. Use the example as a reference to generate code. Use getTimeout() as in the example where applicable."
    egs2 = read_examples(typ='glue')
    glue_resp = get_openai_response(system_msg=glue_msg2, prompt=feature_resp, examples=egs2, temp=temp)
    return glue_resp

def get_feature_glue(tc_json_list, temp=0):
    feature_resp = get_feature(tc_json_list=tc_json_list, temp=temp)
    glue_resp = get_glue(feature_resp=feature_resp, temp=temp)
    return feature_resp, glue_resp

if __name__ == "__main__":
    import time
    user_story = "As a user, I want to be able to log in to the Sauce Demo website, add items to my cart, and successfully complete the checkout process so that I can purchase products efficiently."
    acceptance_criteria = '''Login Functionality, Given I am on the Sauce Demo login page, when I enter valid credentials (username and password), and click the login button, then I should be redirected to the inventory page. Given I am on the Sauce Demo login page, when I enter invalid credentials, and click the login button, then I should receive an error message indicating that the login failed. Adding Items to Cart, Given I am logged into the Sauce Demo website, when I navigate to the inventory page, then I should see a list of available products. Given I am on the inventory page, when I click on the "Add to Cart" button for a product, then the product should be added to my cart. Given I have added items to my cart, when I view my cart, then I should see all the products I have added. Checkout Process, Given I have items in my cart, when I click on the cart icon and proceed to checkout, then I should be directed to the checkout page. Given I am on the checkout page, when I enter valid shipping information (name, address, etc.), and click on the "Continue to Payment" button, then I should be directed to the payment page. Given I am on the payment page, when I enter valid payment information (card number, expiration date, etc.), and click on the "Place Order" button, then the order should be successfully placed. Given I have successfully placed an order, then I should receive an order confirmation message. Error Handling, Given I am attempting to log in, add items to my cart, or complete the checkout process, when I encounter any errors (e.g., network issues, server errors, validation errors), then appropriate error messages should be displayed, guiding me on how to proceed. Session Management, Given I have logged in to the Sauce Demo website, when I navigate to other pages or refresh the page, then my login session should persist unless I explicitly log out. Validation, Given I am providing input in forms (e.g., login, checkout), appropriate validation should be performed (e.g., valid email format, required fields) and error messages should be displayed if validation fails. By fulfilling the above acceptance criteria, the user should be able to smoothly log in, add items to their cart, complete the checkout process, and make a successful purchase.'''

    tc_json = generate_test_cases_json(user_story, acceptance_criteria, temp=0)
    feature_resp, glue_resp = get_feature_glue(tc_json_list=tc_json, temp=0)
    
    print("Generated Feature:")
    print(feature_resp)
    print("\nGenerated Glue:")
    print(glue_resp)
