import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import joblib

# Load model and label binarizer
with tf.device('/CPU:0'):
    model = tf.keras.models.load_model('/home/george/Desktop/Intent_Recognition/intent_recognition_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    label_binarizer = joblib.load('/home/george/Desktop/Intent_Recognition/label_binarizer.pkl')

# Preprocessing function for input text
def preprocess_input(texts):
    
    if isinstance(texts, str): 
        texts = [texts]
    return tf.convert_to_tensor(texts, dtype=tf.string)

# Prediction function
def predict_intent(texts, threshold=0.5):

    processed_texts = preprocess_input(texts)
    predictions = model(processed_texts)
    probabilities = tf.nn.softmax(predictions).numpy()

    # Get all possible class labels
    class_labels = label_binarizer.classes_.tolist()

    predicted_classes = []
    probs_with_functions = []

    for prob in probabilities:
        max_prob = max(prob)  # Get the maximum probability for the current input
        if max_prob < threshold:
            # Assign "other" if no class probability exceeds the threshold
            predicted_classes.append("other")
        else:
            # Get the class with the highest probability
            predicted_class = class_labels[prob.argmax()]
            predicted_classes.append(predicted_class)
        
        # Map probabilities to class labels
        prob_with_function = dict(zip(class_labels, prob.tolist()))
        probs_with_functions.append(prob_with_function)

    return predicted_classes, probs_with_functions

# Example usage
if __name__ == "__main__":
    while True:
        # Ask the user for input
        user_input = input("Enter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        # Get predictions and probabilities
        intents, probabilities = predict_intent(user_input, threshold=0.6)

        # Display the results
        print(f"Input: {user_input}")
        print(f"Predicted Intent: {intents[0]}")
        print(f"Class Probabilities: {probabilities[0]}")
        print("-" * 50)
