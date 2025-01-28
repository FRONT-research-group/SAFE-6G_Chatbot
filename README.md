# SAFE-6G_Chatbot
Repository of SAFE-6G Chatbot

![image](https://github.com/user-attachments/assets/a045aae1-70a0-4f29-b5c2-328e04be5690)


More details can be found in this [link](https://telefonicacorp.sharepoint.com/:p:/r/sites/SAFE-6G-SNS-2023.TMELA/Shared%20Documents/WP3/3.%20Tasks/%CE%A43.5/SAFE-6G_NCSRD_WP3_Chatbot.pptx?d=w299bf271ebd9473690ed9b85a9de2576&csf=1&web=1&e=VN2UjE)



**Intent Recognition Model:
1. Clone the Repository
2. Create a Virtual Environment
  
  python -m venv env
  source env/bin/activate  # On macOS/Linux
  env\Scripts\activate     # On Windows

4. Install Dependencies
  pip install -r requirements.txt

**Usage Instructions
**1. Run the Model
To use the intent recognition model, you can run the Small_bert_intent.py file 
  python Small_bert_intent.py


**Running Inference
To predict intents using the pre-trained model, run the predict_intent.py script. The script will ask for user input and return the predicted intent and class probabilities.
  python predict_intent.py

