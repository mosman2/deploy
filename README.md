
# Project Setup

## Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/deploy.git
   cd deploy
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv myenv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a `.env` file and add your OpenAI API key:**
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key" > .env
   ```

## Extract `requirements.txt`

1. **Generate `requirements.txt`:**
   ```bash
   pip freeze > requirements.txt
   ```

## Install `requirements.txt`

1. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

## .gitignore

```bash
# Ignore environment variables
.env

# Ignore repository folder
example_data/test_repo/

# Ignore virtual environment
myenv/

