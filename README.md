# RAG Based

## Introduction

This Document Q&A System is an AI-powered application that allows users to ask questions about a specific document and receive accurate, context-aware answers. The system processes a document, creates embedding for chunks of text, stores them in a vector database, and uses a combination of semantic search and a large language model to generate responses.



## Features

- Document processing and embedding creation.
- Hugging Face Sentence Transformers: For generating text embedding
- Semantic search using ChromaDB vector database.
- Query expansion for comprehensive answers. 
- AI-generated responses using Google's Gemini-Pro model.
- User-friendly interface with Gradio and Fast API Backend.
  
  

#### Steps:

- Get your *Hugging face Api Key* and replace it in the code file 

- Get your *Google AI studio Api Key* and replace it in the code file

- Update the file path variable in the code with the path of your text file.

- Run the program using:
  
  ```
  $ python3 app.py 
  ```

Head over to the URL: http://localhost:8000/ to ask and test.



**NOTE:**

.ODT file is being used here

 
