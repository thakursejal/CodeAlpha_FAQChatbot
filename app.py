import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CODEALPHA AI & ML FAQ CHATBOT
# ============================================================

faq_data = [
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial Intelligence (AI) is a field of computer science that enables machines to perform tasks that normally require human intelligence."
    },
    {
        "question": "What is AI used for?",
        "answer": "AI is used for prediction, recommendation, language processing, image analysis, automation, and decision support."
    },
    {
        "question": "How does artificial intelligence work?",
        "answer": "AI systems process data, identify patterns, and use those patterns to make predictions, decisions, or generate outputs."
    },
    {
        "question": "What are the advantages of AI?",
        "answer": "AI can automate repetitive tasks, analyze large amounts of data, improve efficiency, and support faster decision-making."
    },
    {
        "question": "What are the limitations of AI?",
        "answer": "AI depends on the quality of its data and models and may produce incorrect results. It requires careful development and evaluation."
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a branch of AI in which computers learn patterns from data and use those patterns to make predictions or decisions."
    },
    {
        "question": "What are the types of machine learning?",
        "answer": "The major types are supervised learning, unsupervised learning, and reinforcement learning."
    },
    {
        "question": "What is supervised learning?",
        "answer": "Supervised learning trains a model using labeled data so that it can learn to make predictions or classifications."
    },
    {
        "question": "What is unsupervised learning?",
        "answer": "Unsupervised learning works with unlabeled data and attempts to discover patterns, structures, or groups within that data."
    },
    {
        "question": "What is reinforcement learning?",
        "answer": "Reinforcement learning is an approach in which an agent learns by interacting with an environment and receiving rewards or penalties."
    },
    {
        "question": "What is classification?",
        "answer": "Classification is a supervised learning task that assigns an input to one of several predefined categories."
    },
    {
        "question": "What is regression?",
        "answer": "Regression is a supervised learning task used to predict continuous numerical values."
    },
    {
        "question": "What is training data?",
        "answer": "Training data is the data used to teach a machine learning model to learn patterns and relationships."
    },
    {
        "question": "What is testing data?",
        "answer": "Testing data is data kept separate from training data and used to evaluate how well the trained model performs."
    },
    {
        "question": "What is deep learning?",
        "answer": "Deep learning is a subset of machine learning that uses multi-layer neural networks to learn complex patterns from data."
    },
    {
        "question": "What is a neural network?",
        "answer": "A neural network is a computational model made of interconnected nodes or neurons that learn patterns from data."
    },
    {
        "question": "What is a CNN?",
        "answer": "CNN stands for Convolutional Neural Network. It is a deep learning architecture commonly used for image and visual data."
    },
    {
        "question": "What is natural language processing?",
        "answer": "Natural Language Processing (NLP) is a field of AI that enables computers to process, understand, and work with human language."
    },
    {
        "question": "What is NLP?",
        "answer": "NLP stands for Natural Language Processing. It allows computers to understand, process, and generate human language."
    },
    {
        "question": "What is computer vision?",
        "answer": "Computer Vision is a field of AI that enables computers to understand and analyze images and videos."
    },
    {
        "question": "What is a chatbot?",
        "answer": "A chatbot is a software application that interacts with users through natural language and provides automated responses."
    },
    {
        "question": "What is a dataset?",
        "answer": "A dataset is a collection of data used to train, test, and evaluate machine learning models."
    },
    {
        "question": "What is an AI model?",
        "answer": "An AI model is a computational model trained on data to recognize patterns and make predictions or generate outputs."
    },
    {
        "question": "What is overfitting?",
        "answer": "Overfitting occurs when a machine learning model learns the training data too closely and performs poorly on new, unseen data."
    },
    {
        "question": "What is underfitting?",
        "answer": "Underfitting occurs when a model is too simple to learn the important patterns in the training data."
    },
    {
        "question": "What is an epoch?",
        "answer": "An epoch is one complete pass through the entire training dataset during model training."
    },
    {
        "question": "What is a feature in machine learning?",
        "answer": "A feature is an individual measurable property or characteristic of the data that is used by a machine learning model."
    },
    {
        "question": "What is a label in machine learning?",
        "answer": "A label is the target or expected output associated with an input in supervised learning."
    },
    {
        "question": "What is precision?",
        "answer": "Precision measures how many of the instances predicted as positive are actually positive."
    },
    {
        "question": "What is recall?",
        "answer": "Recall measures how many of the actual positive instances are correctly identified by the model."
    },
    {
        "question": "What is a confusion matrix?",
        "answer": "A confusion matrix is a table used to evaluate a classification model by showing correct and incorrect predictions for different classes."
    },
    {
        "question": "What is an activation function?",
        "answer": "An activation function determines the output of a neuron in a neural network and helps the network learn complex patterns."
    },
    {
        "question": "What is backpropagation?",
        "answer": "Backpropagation is an algorithm used to train neural networks by calculating errors and updating model weights to reduce the error."
    },
    {
        "question": "What is LSTM?",
        "answer": "LSTM stands for Long Short-Term Memory. It is a type of recurrent neural network designed to learn long-term dependencies in sequential data."
    },
    {
        "question": "What is generative AI?",
        "answer": "Generative AI is a type of artificial intelligence that can create new content such as text, images, audio, or code."
    },
    {
        "question": "What is a large language model?",
        "answer": "A large language model is an AI model trained on large amounts of text data to understand and generate human-like language."
    },
    {
        "question": "What is TF-IDF?",
        "answer": "TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a technique used to represent text numerically based on the importance of words."
    },
    {
        "question": "What is cosine similarity?",
        "answer": "Cosine similarity measures how similar two text vectors are by calculating the cosine of the angle between them."
    },
    {
    "question": "What is an AI algorithm?",
    "answer": "An AI algorithm is a set of computational steps used to process data, identify patterns, and make predictions or decisions."
    },
    {
    "question": "What is data preprocessing?",
    "answer": "Data preprocessing is the process of cleaning, transforming, and preparing raw data so that it can be effectively used by a machine learning model."
    }
]


# ============================================================
# PREPARE FAQ MODEL
# ============================================================

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

faq_vectors = vectorizer.fit_transform(questions)


# ============================================================
# CHATBOT FUNCTION
# ============================================================

def chatbot(user_question):

    if not user_question or not user_question.strip():
        return "Please enter a question so I can help you. 😊"

    user_question = user_question.strip()

    # Convert user question into TF-IDF vector
    user_vector = vectorizer.transform([user_question])

    # Calculate similarity with all FAQ questions
    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    # Find the most similar question
    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    # Confidence threshold
    if best_score < 0.25:
        return (
            "Sorry, I couldn't find a suitable answer to that question. "
            "Please try asking something related to Artificial Intelligence "
            "or Machine Learning."
        )

    return answers[best_match_index]


# ============================================================
# GRADIO INTERFACE
# ============================================================

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        label="💬 Your Question",
        placeholder="Ask me anything about AI & Machine Learning...",
        lines=2
    ),
    outputs=gr.Textbox(
        label="🤖 Chatbot Answer",
        lines=5
    ),
    title="🤖 CodeAlpha AI & ML FAQ Chatbot",
    description=(
        "NLP-based FAQ chatbot using TF-IDF and Cosine Similarity"
    ),
    examples=[
        ["What is artificial intelligence?"],
        ["What is machine learning?"],
        ["What is NLP?"],
        ["What is TF-IDF?"],
        ["What is computer vision?"],
        ["What is a chatbot?"]
    ]
)


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()
