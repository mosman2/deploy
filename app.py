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
    loader = GitLoader(
        clone_url=repo_url,
        repo_path=repo_path,
        branch=branch,
    )
    return loader.load()

# Function to read a specific Jupyter notebook file from the repository
def read_notebook_file(repo_path, notebook_file):
    notebook_path = os.path.join(repo_path, notebook_file)
    with open(notebook_path, "r") as f:
        notebook_content = json.load(f)
    return notebook_content

# Function to extract and combine code cells from the notebook
def extract_and_combine_code_cells(notebook_content):
    combined_code = ""
    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'code':
            combined_code += ''.join(cell['source']) + "\n"
    return combined_code

# Main function
def main():
    # Load the repository
    load_repository_data(REPO_URL, REPO_PATH, BRANCH)
    # Read the specific notebook file
    notebook_content = read_notebook_file(REPO_PATH, NOTEBOOK_FILE)
    # Extract and combine code cells
    combined_code = extract_and_combine_code_cells(notebook_content)
    
    # Define Pydantic parser
    parser = PydanticOutputParser(pydantic_object=CategorizedCode)
    
    # Prompt template for parsing
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Parse and modularize the code and logically seperate into functions with; dependencies, inputs, model and model parameters, inference code such that the model is ready to be deployed. Return the output as JSON:\n{format_instructions}"),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    
    # Set up the model
    model = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=os.getenv('OPENAI_API_KEY'),
    )

    # Define the chain
    chain = prompt_template | model | parser

    # Invoke the chain
    try:
        parsed_output = chain.invoke({"query": combined_code})
        print(json.dumps(parsed_output.dict(), indent=4))
    except Exception as e:
        print(f"Error parsing response: {e}")

if __name__ == "__main__":
    main()
