# <<< OK FOR DEMO >>>
# <<< LAST UPDATED ON 07-May-2024>>>
# <<< ERROR IS FIXED AND THIS IS ABLE TO HANDLE OUT OF CONTEXT QUESTIONS >>>

import streamlit as st
import os
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import psycopg2


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load PDF documents 
loader = PyPDFDirectoryLoader("data")
data = loader.load()

# text_splitter 
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=750,
    chunk_overlap=50
)

#text_chunks
text_chunks = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()

# converting text_chunks to embeddings and store it into a Vectordb
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

llm = OpenAI(temperature=0.5, max_tokens=1000)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k":2}))

# Define Streamlit UI
st.title("HR Chat Bot")

# function for random question
def random_question():
        st.sidebar.subheader("Sample Questions / FAQs")
        prompt3 = f"""give me 10 HR policy related questions
                    like 
                    " What is the dress code for men in office"
                    "What is the general travel policy"
                    "What are the leaves with out pay cut"
                    "how many  days of leave should I able to take as maternity leave"."""
        random_question = get_answer2(prompt3)["result"]
        st.sidebar.write(f"{random_question}")

# Function to get answer
def get_answer(question, prompt):

    prompt5 = f""" you are an analyst please check whether this question {question} related to this {contexts} context or not, if yes return "TRUE", else "False". """
    TF = llm.invoke(prompt5)
    if TF :
        print(TF)
        question_with_prompt = f"{prompt} {question}"
        answer = qa.invoke(question_with_prompt)
        return answer
    else:
        st.write("PLEASE ASK QUESTION RELATED TO HR POLICIES")
        return None, None  # Returning None for both answer and TF if the question is not HR-related
        


def get_answer2(question2):
    answer = llm.invoke(question2)
    return {"result": answer}

# Chat Bot UI
st.subheader("Ask your question here:")

question = st.text_input("example")

if st.button("Get Answer"):
    if question:
        contexts = ["leaves", "reimbursement", "travel", "dress_code", "maintenance_costs","food and beverages","goal setting","shift allowances","work timing","attendance","wfh policy","performance appraisal process", "compensation revision policy","ratings","Maternity Leave"]
        prompt4 = f""" you are a good question analyzer, by analyzing this {question}, find that in which of the {contexts} it belongs to, give me only the context {contexts} from this.Give the best top most context I want only 1 context as a single word .If the {question} is not matching any of the given {contexts} context then give me this one word "out of context" .""" 
        context = get_answer(question,prompt4)
        
        # Database connection details
        dbname = "testDB"
        user = "postgres"
        password = "Password"
        host = "localhost"  
        port = "5432"  

        # Function to connect to the database
        def connect_to_database():
            try:
                connection = psycopg2.connect(
                    dbname=dbname, user=user, password=password, host=host, port=port
                )
                cursor = connection.cursor()
                print("Connected to the database")
                return connection, cursor
            except Exception as e:
                print("Unable to connect to the database:", e)
                return None, None

        # Function to create the table for questions and answers
        def create_qa_table(cursor):
            try:
                create_table_query = '''
                    CREATE TABLE IF NOT EXISTS HRTABLE11 (
                        id SERIAL PRIMARY KEY,
                        question TEXT,
                        context TEXT
                    )
                '''
                cursor.execute(create_table_query)
                cursor.connection.commit()
                print("Table created successfully")
            except Exception as e:
                print("Error creating table:", e)

        # Function to add a question and its context to the table
        def add_qa(cursor, question1, context1):
            try:
                insert_query = "INSERT INTO HRTABLE11 (question, context) VALUES (%s, %s)"
                cursor.execute(insert_query, (question1, context1))
                cursor.connection.commit()
                print("Question and context added successfully")
            except Exception as e:
                print("Error adding question and context:", e)

        connection, cursor = connect_to_database()
        if connection and cursor:
            create_qa_table(cursor)
            question1 = str(question)
            context1 = str(context["result"])  # Storing the result from the language model
            add_qa(cursor, question1, context1)
            cursor.close()
            connection.close()
                
        prompt =""" **You are an useful HRchat bot so don't hallucinate**
                **If the user asks out of context questions then answer " This question is out of context"**.
                **If the user uses cuss words then answer  " Such Derogatory words are not entertained "**.
                Carefully assess the query posed by the user, noting any implicit or explicit queries.
                Query the designated retrieval database for accurate and contextually relevant data.
                Wrap up by formulating a response that conforms to the expected structure and format, presenting the information in a clear and engaging manner.
                provide answers point wise in the form of bullets. Please show only the top bullet points for the answer which is highly related to the question.just give me most relevent points as answer it should be less than 5 points 
                """

        answer = get_answer(question, prompt)
        x = question
        prompt2 = f" Carefully assess the the {x}. please generate 10 questions similar to {x},one by one\n\n"
        answer2 = get_answer2(prompt2)

        
        
        st.write(answer["result"])
        if  "out of context" not in answer["result"]:
            st.sidebar.write("Sample Questions / FAQs\n\n", answer2["result"])
        else:
            random_question()


        
    else:
        random_question()
        st.write("Please input a question first.")

if not question:
    random_question()
