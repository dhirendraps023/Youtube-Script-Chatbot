import os
from dotenv import load_dotenv

# Load OpenAI API key from .env (avoid committing secrets)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY must be set in environment or .env file")
os.environ["OPENAI_API_KEY"] = openai_key

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Fetch the transcript of a YouTube video
vedio_id = "X0btK9X0Xnk"  # Replace with your YouTube video ID  
try:
    transcript = YouTubeTranscriptApi().fetch(vedio_id, languages=['hi'] )
    transcript_text = " ".join([entry.text for entry in transcript])
    #print(transcript_text)  # Print the transcript text
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
except Exception as e:
    print(f"An error occurred: {e}")    

# Split the transcript into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
chunks = text_splitter.split_text(transcript_text)
#print(f"Number of chunks: {len(chunks)}")  # Print the number of chunks

# Create embeddings and store in FAISS
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(chunks, embeddings)
#print(vector_store.index_to_docstore_id)

# Create a prompt template for the chatbot
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="You are a helpful assistant. Answer the following question " \
    "based on the video transcript and say don't know if you don't find " \
    "an answer in the context: {context}, question: {question}"
)

# Create a chatbot function contain retrieval and response generation using the OpenAI language model   
def chatbot(question):
    # Retrieve relevant chunks from the vector store
    relevant_chunks = vector_store.similarity_search(question, k=3)
    
    # Combine the relevant chunks into a single context
    context = " ".join([chunk.page_content for chunk in relevant_chunks])
    
    # Create a prompt with the question and context
    prompt = prompt_template.format(question=question, context=context)
    
    # Generate a response using the OpenAI language model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7 )
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    return response.content

# Example usage
user_question = input("Ask a question about the video: ")
answer = chatbot(user_question)
print(f"Question: {user_question}") 
print(f"Answer: {answer}")  
