import PyPDF2
import re
import sys

# Function to read PDF document and extract text
def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to build chatbot logic
def chatbot(document_text):
    while True:
        question = input("You: ").lower()
        if question == 'exit':
            print("Chatbot: Goodbye!")
            break
        # Preprocess the question for matching
        question = preprocess_text(question)
        if question in document_text:
            # If the question is found in the document, provide the answer
            print("Chatbot: Yes, I found information related to your question.")
        else:
            # If the question is not found, provide a default response
            print("Chatbot: I don't know the answer.")

# Main function
def main(file_path):
    # Read the document and extract text
    if file_path.endswith('.pdf'):
        document_text = read_pdf(file_path)
    else:
        print("Unsupported file format.")
        return
    # Preprocess the document text
    document_text = preprocess_text(document_text)
    # Start the chatbot
    chatbot(document_text)

if __name__ == "__main__":
    file_path = sys.argv[1]
    main(file_path)
