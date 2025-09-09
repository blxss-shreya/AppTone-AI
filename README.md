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

## Setup Instructions

1. Create a dedicated folder for the project and navigate into it:
- mkdir ~/Projects
- cd ~/Projects

2. Clone the repository and navigate into the project folder:

- git clone https://github.com/blxss-shreya/AppTone-AI.git
- cd AppTone-AI

3. Activate the virtual environment:

- source venv/bin/activate

4. Install dependencies:

- pip install -r requirements.txt

5. Run the application:

- python main.py

Once running, open http://127.0.0.1:5000 in your browser.

## Notes
![landing](images/screenshot_1.png)
![Diabetes Prediction](images/screenshot_2.png)
![Sample Model Output](images/screenshot_3.png)

