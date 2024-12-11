from pinecone import Pinecone, ServerlessSpec
# from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

class RAGChatbot:
    load_dotenv()
    def __init__(self, index_name="langchain-fdemo", namespace="english"):
        PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.namespace = namespace
        self.embeddings = HuggingFaceEmbeddings(
            model_name="neuml/pubmedbert-base-embeddings"
        )
        self.initialize_index()
        self.initialize_qa_chain()

    def initialize_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.docsearch = PineconeVectorStore.from_existing_index(
            self.index_name,
            self.embeddings,
            namespace=self.namespace
        )

    def initialize_qa_chain(self):
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm = HuggingFaceEndpoint(
            repo_id="tiiuae/falcon-7b-instruct", max_length=500, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
        )
        prompt_template = """You are a friendly and knowledgeable yoga assistant. Your task is to help users by detecting their mood based on their input and providing appropriate yoga routine advice. 
        1. Analyze the user's message to determine their emotional state (e.g., happy, sad, anxious, stressed, etc.).
        2. Based on the detected mood, suggest a proper yoga routine with time length that can help improve their emotional state. 
        3. Provide a brief explanation of how each suggested yoga pose can benefit the user in relation to their mood.
        Respond to their inquiries in a helpful manner. If you don't know the answer, 
        kindly apologize and let the user know.
        
        Context: {context}
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.docsearch.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

    def chat(self, query):
        return self.qa.invoke(query)
    
# RAGChatbot().chat(query="How can I be happy?")

# # testing
# bot = RAGChatbot()
# input = input("Ask me anything: ")
# result = bot.chat(query=input)["result"]
# print(result)