import streamlit as st
import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('all')
# Initialiser le lemmatizer
lemmatizer = WordNetLemmatizer()

# Charger les fichiers essentiels pour le chatbot
model = load_model('model/chatbot_model.h5')

# Charger les intents (fichier JSON avec les patterns, tags, responses, etc.)
with open('data/intents.json') as file:
    intents = json.load(file)

# Charger les données de classes et de mots (issues du prétraitement)
words = np.load('save/words.pkl', allow_pickle=True)
classes = np.load('save/classes.pkl', allow_pickle=True)

# Fonction pour prétraiter les phrases utilisateur
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convertir une phrase en sac de mots
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Fonction pour obtenir une réponse à partir du modèle
def predict_class(sentence, model):
    bow_input = bow(sentence, words)
    bow_input = np.array([bow_input])
    prediction = model.predict(bow_input)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, res] for i, res in enumerate(prediction) if res > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            break
    return response

# Interface Streamlit
st.title("Chatbot Tourisme au Gabon")

# Stocker l'historique des discussions dans une session Streamlit
if 'history' not in st.session_state:
    st.session_state.history = []

# Fonction pour afficher la discussion dans un style de chat
def display_chat():
    chat_history = ""
    for message, sender in st.session_state.history:
        if sender == 'user':
            # Afficher le message de l'utilisateur dans une bulle verte
            chat_history += f"""
            <div style='text-align: right;'>
                <div style='background-color: #dcf8c6; border-radius: 10px; padding: 10px; margin: 5px; display: inline-block; max-width: 70%; word-wrap: break-word;'>
                    <strong>Vous:</strong> {message}
                </div>
            </div>
            """
        else:
            # Afficher le message du bot dans une bulle grise
            chat_history += f"""
            <div style='text-align: left;'>
                <div style='background-color: #f1f0f0; border-radius: 10px; padding: 10px; margin: 5px; display: inline-block; max-width: 70%; word-wrap: break-word;'>
                    <strong>Bot:</strong> {message}
                </div>
            </div>
            """
    st.markdown(chat_history, unsafe_allow_html=True)

# Demander l'entrée de l'utilisateur
user_input = st.text_input("Vous: ", "Tapez votre message ici...")

# Si l'utilisateur saisit une question
if user_input:
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.history.append((user_input, 'user'))
    
    # Prédire l'intention de l'utilisateur
    ints = predict_class(user_input, model)
    
    # Obtenir la réponse correspondante
    res = get_response(ints, intents)
    
    # Ajouter la réponse du bot à l'historique
    st.session_state.history.append((res, 'bot'))

# Afficher la discussion comme un chat WhatsApp
display_chat()
