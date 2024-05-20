import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Define the Pydantic model for the expected JSON structure
class CategorizedCode(BaseModel):
    dependencies: list[str] = Field(description="List of dependencies")
    inputs: list[str] = Field(description="List of inputs")
    models: list[str] = Field(description="List of models and model parameters")
    inference_code: list[str] = Field(description="List of inference code snippets")
    functions: list[str] = Field(description="List of functions")

# Inputs
REPO_URL = "https://github.com/yoonholee/edward2-notebooks"
REPO_PATH = "./example_data/test_repo"
BRANCH = "master"
NOTEBOOK_FILE = "0-2. Shapes.ipynb"  # Focus on this file

# Function to load data from a Git repository
def load_repository_data(repo_url, repo_path, branch):
    try:
        loader = GitLoader(
            clone_url=repo_url,
            repo_path=repo_path,
            branch=branch,
        )
        loader.load()
        print(f"Repository {repo_url} successfully loaded.")
    except Exception as e:
        print(f"Error loading repository: {e}")

# Function to read a specific Jupyter notebook file from the repository
def read_notebook_file(repo_path, notebook_file):
    notebook_path = os.path.join(repo_path, notebook_file)
    try:
        with open(notebook_path, "r") as f:
            notebook_content = json.load(f)
        print(f"Notebook {notebook_file} successfully read.")
        return notebook_content
    except FileNotFoundError:
        print(f"Notebook file {notebook_file} not found at path {repo_path}.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the notebook file {notebook_file}.")
    except Exception as e:
        print(f"Error reading notebook file: {e}")

# Function to extract and combine code cells from the notebook
def extract_and_combine_code_cells(notebook_content):
    try:
        combined_code = ""
        for cell in notebook_content['cells']:
            if cell['cell_type'] == 'code':
                combined_code += ''.join(cell['source']) + "\n"
        print("Code cells successfully extracted and combined.")
        return combined_code
    except KeyError:
        print("Invalid notebook structure; 'cells' key not found.")
    except Exception as e:
        print(f"Error extracting code cells: {e}")

# Function to categorize the combined code
def categorize_code(combined_code):
    parser = PydanticOutputParser(pydantic_object=CategorizedCode)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Parse and modularize the code and logically separate it into functions with; dependencies, inputs, models and model parameters, inference code such that the model is ready to be deployed. Return the output as JSON:\n{format_instructions}"),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    model = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=os.getenv('OPENAI_API_KEY'),
    )

    chain = prompt_template | model | parser

    try:
        parsed_output = chain.invoke({"query": combined_code})
        print("Code successfully categorized.")
        return parsed_output.dict()
    except Exception as e:
        print(f"Error parsing response: {e}")

# Function to generate a deployable Python script from the categorized code
def generate_deployable_script(categorized_code):
    try:
        script_content = "# Auto-generated deployable script\n\n"

        # Add dependencies
        if categorized_code.get('dependencies'):
            script_content += "# Dependencies\n"
            for dependency in categorized_code['dependencies']:
                script_content += f"import {dependency}\n"
            script_content += "\n"

        # Add input variables
        if categorized_code.get('inputs'):
            script_content += "# Inputs\n"
            for input_var in categorized_code['inputs']:
                script_content += f"{input_var} = None  # Replace with actual input\n"
            script_content += "\n"

        # Add model and model parameters
        if categorized_code.get('models'):
            script_content += "# Models and Model Parameters\n"
            for model in categorized_code['models']:
                script_content += f"{model}\n"
            script_content += "\n"

        # Add functions
        if categorized_code.get('functions'):
            script_content += "# Functions\n"
            for function in categorized_code['functions']:
                script_content += f"{function}\n"
            script_content += "\n"

        # Add inference code
        if categorized_code.get('inference_code'):
            script_content += "# Inference Code\n"
            for code in categorized_code['inference_code']:
                script_content += f"{code}\n"
            script_content += "\n"

        return script_content
    except Exception as e:
        print(f"Error generating deployable script: {e}")

# Function to save the generated script to a file
def save_script_to_file(script_content, file_path):
    try:
        with open(file_path, "w") as script_file:
            script_file.write(script_content)
        print(f"Script successfully saved to {file_path}.")
    except Exception as e:
        print(f"Error saving script to file: {e}")

# Main function
def main():
    load_repository_data(REPO_URL, REPO_PATH, BRANCH)
    notebook_content = read_notebook_file(REPO_PATH, NOTEBOOK_FILE)
    if notebook_content:
        combined_code = extract_and_combine_code_cells(notebook_content)
        if combined_code:
            categorized_code = categorize_code(combined_code)
            if categorized_code:
                script_content = generate_deployable_script(categorized_code)
                if script_content:
                    save_script_to_file(script_content, "deployable_script.py")

if __name__ == "__main__":
    main()
