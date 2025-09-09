# AppTone AI

AppTone AI is a machine learning-based web application that analyzes app reviews and provides overall sentiment insights. It uses NLP models (VADER and RoBERTa) to evaluate reviews and display both overall sentiment and sample reviews.

## Features
- Scrapes app reviews from Google Play Store
- Performs sentiment analysis using VADER and RoBERTa
- Provides confidence thresholds for more accurate detection of positive, neutral, and negative reviews
- Simple web interface for entering app data and viewing results

## Technologies Used
- Python, Flask
- HTML, CSS, JavaScript
- NLTK, HuggingFace Transformers
- pandas, NumPy, scikit-learn
- REST API for backend communication

# AppTone AI

## Setup Instructions

1. Create a dedicated folder for the project:
   
```bash
mkdir ~/Projects
cd ~/Projects

2. Clone the repository:

git clone https://github.com/blxss-shreya/AppTone-AI.git
cd AppTone-AI

3. Activate the virtual environment:

source venv/bin/activate

4. Install dependencies:

pip install -r requirements.txt

5. Run the application:

python main.py



