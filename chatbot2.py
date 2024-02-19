import PyPDF2
import re
import nltk
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
from pinecone import Pinecone
import sys


nltk.download('punkt')
# Function to read PDF document and extract paragraphs
def read_pdf(file_path):
    paragraphs = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page_text = reader.pages[page_num].extract_text()
            paragraphs.extend(page_text.split('\n\n'))  # Split by double newline for paragraphs
    return paragraphs

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to tokenize and preprocess paragraphs
def tokenize_paragraphs(paragraphs):
    tokenized_paragraphs = []
    for para in paragraphs:
        tokens = nltk.word_tokenize(para)
        tokens = [token.lower() for token in tokens if token.isalnum()]
        tokenized_paragraphs.append(tokens)
    return tokenized_paragraphs

# Function to create paragraph embeddings
def create_embeddings(paragraphs):
    dictionary = Dictionary(paragraphs)
    corpus = [dictionary.doc2bow(para) for para in paragraphs]
    tfidf = TfidfModel(corpus)
    index = SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
    return dictionary, tfidf, index

# Function to perform semantic similarity search
def semantic_similarity_search(question, dictionary, tfidf, index, top_k=3):
    tokens = nltk.word_tokenize(question)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    query_bow = dictionary.doc2bow(tokens)
    query_tfidf = tfidf[query_bow]
    sims = index[query_tfidf]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])[:top_k]
    return sims

# Main function
def main(file_path):
    # Read the document and extract paragraphs
    if file_path.endswith('.pdf'):
        paragraphs = read_pdf(file_path)
    else:
        print("Unsupported file format.")
        return
    # Preprocess the paragraphs
    preprocessed_paragraphs = [preprocess_text(para) for para in paragraphs]
    tokenized_paragraphs = tokenize_paragraphs(preprocessed_paragraphs)
    # Create embeddings and index for semantic similarity search
    dictionary, tfidf, index = create_embeddings(tokenized_paragraphs)
    # Connect to Pinecone vector database
    pinecone = Pinecone(api_key='5830a444-8928-4068-969b-f72f7e403508')
    # Create or retrieve Pinecone index
    pinecone.create_index('document_index',dimension=300,spec={
    "method": "hnsw",
    "parameters": {
        "M": 16,
        "ef_construction": 100,
        "ef_search": 128
    }
})
    pinecone_index = pinecone.index('document_index')
    # Insert paragraphs into Pinecone index
    pinecone_index.upsert(ids=range(len(paragraphs)), vectors=tfidf[corpus])
    # Interact with the chatbot
    while True:
        question = input("You: ").lower()
        if question == 'exit':
            print("Chatbot: Goodbye!")
            break
        # Perform semantic similarity search
        similar_paragraphs = semantic_similarity_search(question, dictionary, tfidf, index)
        for para_index, _ in similar_paragraphs:
            print(f"Chatbot: Here is a relevant paragraph:\n{paragraphs[para_index]}")

if __name__ == "__main__":
    file_path = sys.argv[1]
    main(file_path)
