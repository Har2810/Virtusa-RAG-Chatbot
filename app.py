import os
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai
from odf.opendocument import load
from odf.text import P
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
app = FastAPI()

# Set up ChromaDB
persist_directory = "./chroma_db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

chroma_client = chromadb.PersistentClient(path=persist_directory)
collection = chroma_client.get_or_create_collection(name="my_collection")

# Set up embedding function
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="hugging_face_api_key",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Set up Gemini
GOOGLE_API_KEY = "google_ai_studio_api_key"
genai.configure(api_key=GOOGLE_API_KEY)

file_path = 'path_to_your_text_file'

def create_chunks(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def process_document(file_path):
    doc = load(file_path)
    content = []
    for paragraph in doc.getElementsByType(P):
        paragraph_text = ''.join(node.data for node in paragraph.childNodes if node.nodeType == 3)
        content.append(paragraph_text.strip())
    document_text = ' '.join(content)
    
    chunks = create_chunks(document_text)
    embeddings = huggingface_ef(chunks)
    if isinstance(embeddings, dict):
        embeddings = embeddings.get('embeddings', [])
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"text": chunk} for chunk in chunks]
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

def query_collection(query_text, n_results=10):
    query_embedding = huggingface_ef([query_text])
    if isinstance(query_embedding, dict):
        query_embedding = query_embedding.get('embeddings', [])
    if isinstance(query_embedding, list) and len(query_embedding) > 0:
        query_embedding = query_embedding[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "embeddings"]
    )
    relevant_chunks = [metadata['text'] for metadata in results['metadatas'][0]]
    chunk_embeddings = results['embeddings'][0]
    similarity_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    return relevant_chunks, similarity_scores.tolist()


def generate_answer(query, contexts, similarity_scores):
    if not contexts or max(similarity_scores) <= 0.1:
        return "Information not found in the document."
    context = "\n\n".join(contexts)
    prompt = f"""Based on the following context, answer the question. If the information is not explicitly mentioned, make reasonable inferences and
clearly state that you are doing so. If the answer cannot be derived from the given context,
state that the information is not found in the document.

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = genai.GenerativeModel('gemini-pro').generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.8,
            top_k=40,
            max_output_tokens=250
        ))
        if response.parts:
            return response.text
        else:
            return "Error: The model did not generate any content."
    except Exception as e:
        return f"Error: Unable to generate response. {str(e)}"

def expand_query(query):
    prompt = f"Given the question: '{query}', generate 2-3 related questions that might help provide a more comprehensive answer. Format the output as a comma-separated list."
    try:
        response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
        if response.parts:
            expanded_queries = [q.strip() for q in response.text.split(',')]
            return [query] + expanded_queries
        else:
            return [query]
    except Exception as e:
        print(f"Error in query expansion: {str(e)}")
        return [query]


class Query(BaseModel):
    question:str

css = """
body {
    font-family: Arial, sans-serif;
}
.container {
    margin: auto;
    padding: 20px;
    border-radius: 10px;
    background-color: #48e8e0;
}
.title {
    text-align: center;
    color: #2c3e50;
}
"""


def gradio_ask(question):
    try:
        expanded_queries = expand_query(question)
        all_chunks = []
        all_scores = []
        for eq in expanded_queries:
            chunks, scores = query_collection(eq, n_results=5)
            all_chunks.extend(chunks)
            all_scores.extend(scores)
        answer = generate_answer(question, all_chunks, all_scores)
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"


with gr.Blocks(theme=gr.themes.Soft(), css=css) as iface:
    gr.Markdown(
    """
    # 📚 Document Q&A System
    
    Ask any question about the Women in Open Source document, 
    and we will answers based on the content.
    
    
    Let's get started!
    """
    )
    
    with gr.Row():
        # with gr.Column(scale=1):
            # gr.Image(logo_path, show_label=False)
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your Question",
                placeholder="Enter your question here...",
                lines=2
            )
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear")
    
    answer = gr.Textbox(label="Generated Answer", lines=5)
    
    submit_btn.click(fn=gradio_ask, inputs=question, outputs=answer)
    clear_btn.click(fn=lambda: "", inputs=None, outputs=question)

# iface = gr.Interface(
#     fn=gradio_ask,
#     inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
#     outputs="text",
#     title="Document Q&A System",
#     description="Ask questions about the document and get AI-generated answers."
# )

# FastAPI endpoint
@app.post("/ask")
async def ask_question(query: Query):
    return {"answer": gradio_ask(query.question)}

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

# Process the document on startup
@app.on_event("startup")
async def startup_event():
    if collection.count() == 0:
        process_document(file_path)
        print("Document processed and embeddings created.")
    else:
        print("Document already processed. Using existing embeddings.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)