import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load OpenAI API key from .env (avoid committing secrets)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY must be set in environment or .env file")
os.environ["OPENAI_API_KEY"] = openai_key

# Fetch the transcript of a YouTube video
vedio_id = "4C_zbU3abfA"  # Replace with your YouTube video ID  
try:
    transcript = YouTubeTranscriptApi().fetch(vedio_id, languages=['en'] )
    transcript_text = " ".join([entry.text for entry in transcript])
    #print(transcript_text)  # Print the transcript text
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
except Exception as e:
    print(f"An error occurred: {e}")    

# Split the transcript into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
chunks = text_splitter.split_text(transcript_text)

# Create embeddings and store in FAISS
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(chunks, embeddings)

# Retrieve relevant chunks from the vector store
retrieve = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

user_question = input("Ask a question about the video: ")

relevant_chunks = retrieve.invoke(user_question)  # Example question to retrieve relevant chunks

# Combine the relevant chunks into a single context
def format_retrieval_output(relevant_chunks):
    return " ".join([chunk.page_content for chunk in relevant_chunks])

print(f"Relevant Chunks: {format_retrieval_output(relevant_chunks)}")  # Print the relevant chunks
parallel_Chain = RunnableParallel({
    'context': retrieve |RunnableLambda(format_retrieval_output),  # Format the retrieved chunks into a context string
    'question': RunnablePassthrough()  # Pass the question through without modification
})

# Create a prompt template for the chatbot
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="You are a helpful assistant. Answer the following question " \
    "based on the video transcript and say don't know if you don't find " \
    "an answer in the context: {context}, question: {question}"
)

string_output_parser = StrOutputParser()

# Generate a response using the OpenAI language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7 )

main_chain = parallel_Chain | prompt_template | llm | string_output_parser

response = main_chain.invoke(user_question)

#Usage
print(f"Question: {user_question}") 
print(f"Answer: {response}")  
