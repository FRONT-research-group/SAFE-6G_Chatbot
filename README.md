# SAFE-6G_Chatbot
Repository of SAFE-6G Chatbot

![image](https://github.com/user-attachments/assets/a045aae1-70a0-4f29-b5c2-328e04be5690)


More details can be found in this [link](https://telefonicacorp.sharepoint.com/:p:/r/sites/SAFE-6G-SNS-2023.TMELA/Shared%20Documents/WP3/3.%20Tasks/%CE%A43.5/SAFE-6G_NCSRD_WP3_Chatbot.pptx?d=w299bf271ebd9473690ed9b85a9de2576&csf=1&web=1&e=VN2UjE)



Intent Recognition Model
This repository contains the code and resources for an Intent Recognition Model built with TensorFlow. The model predicts the intent of input text and provides probabilities for various predefined classes.

Getting Started
Follow the instructions below to set up and run the project.

1. Clone the Repository
First, clone the repository to your local machine:

bash
Copy
Edit
git clone <repository-url>
cd <repository-folder>
2. Create a Virtual Environment
Set up a Python virtual environment to isolate dependencies:

macOS/Linux:
bash
Copy
Edit
python -m venv env
source env/bin/activate
Windows:
bash
Copy
Edit
python -m venv env
env\Scripts\activate
3. Install Dependencies
Install the required Python libraries listed in requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt
Usage Instructions
1. Run the Model
To initialize and run the model, execute the Small_bert_intent.py script:

bash
Copy
Edit
python Small_bert_intent.py
2. Running Inference
To use the pre-trained model for inference:

Run the predict_intent.py script:
bash
Copy
Edit
python predict_intent.py
Enter a sentence when prompted.
The script will output the predicted intent and class probabilities.
