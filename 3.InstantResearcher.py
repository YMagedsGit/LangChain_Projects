from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the Data
loader = TextLoader("84.txt.utf-8")
docs = loader.load()

# Split the Data
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)

# Embeddings + VectorStore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based only on the context:\n\n{context}"),
    ("human", "{input}")
])

# Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# RAG chain (LCEL style)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": (lambda x: x["input"]) | retriever | format_docs,
        "input": lambda x: x["input"]
    }
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke({"input": "Who is the Creature?"})
print(response)
