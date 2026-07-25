#support ticket classifier
import pandas as pd
df=pd.read_csv("support_tickets_labeled(1).csv")
print(df.shape)

print(df.head())
print(df['category'].value_counts())

import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    stopwords = {"a","an","the","and","or","is","are","to","of","in","on",
                 "for","with","this","that","i","my","you","your","please"}
    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words)
df["text"] = (df["subject"] + " " + df["body"]).apply(clean_text)
print(df["text"].iloc[0])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X=vectorizer.fit_transform(df["text"])
print(X.shape)


from sklearn.model_selection import train_test_split
y=df["category"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
print(X_train.shape,X_test.shape)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model trained!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

def classify_ticket(subject, body):
    text=clean_text(subject+""+body)
    vector=vectorizer.transform([text])
    prediction=model.predict(vector)[0]
    return prediction
result=classify_ticket("App keeps crashing","I am unable to use the app as it crashes every time I open it.")
print(result)

def classify_ticket(subject,body):
    text=clean_text(subject+""+body)
    vector=vectorizer.transform([text])
    probs=model.predict_proba(vector)[0]
    best_index=probs.argmax()
    category=model.classes_[best_index]
    confidence=round(probs[best_index]*100,1)
    return category, confidence
category, confidence=classify_ticket("Refund request","I would like to request a refund for my recent purchase.")
print(category, confidence, "%")

def classify_ticket(subject, body):
    text = clean_text(subject + " " + body)
    vector = vectorizer.transform([text])
    probs = model.predict_proba(vector)[0]
    best_index = probs.argmax()
    category = model.classes_[best_index]
    confidence = round(probs[best_index] * 100, 1)
    needs_review = confidence < 60
    return category, confidence, needs_review
category, confidence, needs_review = classify_ticket("banana purple seventeen", "")
print(category, confidence, needs_review)

new_tickets = [
    ("Overcharged this month", "You billed me twice for my subscription"),
    ("App won't open", "It crashes immediately every time I launch it"),
    ("How many vacation days left", "I want to check my remaining PTO for this year"),
    ("Do you have student pricing", "Is there a discount for university students"),
    ("Great job on the update", "The new dashboard looks fantastic, thank you"),
]
for subject, body in new_tickets:
    category, confidence, needs_review = classify_ticket(subject, body)
    print(f"{subject} -> {category} ({confidence}%) | Needs review: {needs_review}")


