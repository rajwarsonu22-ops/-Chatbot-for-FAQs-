import json

from preprocess import preprocess

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ
with open("faq_data.json", "r") as file:
    faq = json.load(file)

questions = [item["question"] for item in faq]
answers = [item["answer"] for item in faq]

# Preprocess questions
processed_questions = [preprocess(q) for q in questions]

# TF-IDF
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(processed_questions)

print("===== FAQ Chatbot =====")

while True:

    user = input("\nYou : ")

    if user.lower() == "exit":
        print("Bot : Goodbye!")
        break

    processed_user = preprocess(user)

    user_vector = vectorizer.transform([processed_user])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()

    score = similarity[0][best_match]

    if score > 0.25:
        print("Bot :", answers[best_match])
    else:
        print("Bot : Sorry, I don't understand your question.")