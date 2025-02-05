import os
import re
import io
import nest_asyncio
import asyncio


from flask import Flask, render_template, request, url_for, redirect, session
from pymongo import MongoClient
from llama_parse import LlamaParse
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.schema import Document
# from langchain_huggingface import HuggingFaceEmbeddings


from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import bcrypt
import time
#set app as a Flask instance 
app = Flask(__name__)
#encryption relies on secret keys so they could be run
app.secret_key = "testing"


# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


# connect to LLAMA CLOUD
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-aju7w6cXsgyFPGEsECuHMQIrp3th95a1lOfOxyuXkohnGKur"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_2rixW1_8cyc4WuwQaqjBbKshbJAz1YdoLLZRks3ku4JkQ7E97ij9gSPCLMiprpQrsJ2GyT"

pinecone_api_key = "pcsk_2rixW1_8cyc4WuwQaqjBbKshbJAz1YdoLLZRks3ku4JkQ7E97ij9gSPCLMiprpQrsJ2GyT"

# #connect to your Mongo DB database
client = MongoClient("mongodb+srv://admin555:admin555@rag.8zbcn.mongodb.net/?retryWrites=true&w=majority&appName=RAG")
db = client.get_database('total_records')
db2 = client.get_database('total_pdfs')
db3 = client.get_database('pdf_chunks')
records = db.register    
pdfs = db2.register
myLove = db3.meta1
#assign URLs to have a particular route 

# Set the upload folder (make sure this folder exists)
UPLOAD_FOLDER = 'uploads'  # Create a folder named 'uploads' in your project directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def chunkByLlama(indexName, file_like, fileName):      
    document = LlamaParse(
                    result_type="text", streaming=True,
    chunk_size=384
                ).load_data(file_like,extra_info={"file_name": fileName})        
    document_text = "\n".join([f"- {index} -\n{page.text}" for index, page in enumerate(document)])
    # Define chunk boundaries
    chunk_boundaries = [
        "หมวดที่ 1 ข้อมูลทั่วไป",
        "ชื่อปริญญาและสาขาวิชา",
        "3. หลักสูตรและอาจารย์ผู้สอน",
        "3.1.3 รายวิชา",
        "(2) หมวดวิชาเฉพาะ",        
        "แสดงตัวอย่างแผนการศึกษา",
        "รายวิชาที่เป็นรหัสวิชาของหลักสูตร"
    ]

    # Define a pattern to split the last chunk further based on `- xx -`
    split_pattern = r'- \d{1,2} -'

    # Splitting logic
    chunks = []
    current_chunk = []

    lines = document_text.split("\n")
    
    for line in lines:
        if any(boundary in line for boundary in chunk_boundaries) and current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            
            # Further split the chunk if it matches the special case
            if "รายวิชาที่เป็นรหัสวิชาของหลักสูตร" in chunk_text:
                sub_chunks = re.split(split_pattern, chunk_text)
                chunks.extend([sub.strip() for sub in sub_chunks if sub.strip()])
            else:
                chunks.append(chunk_text)

            current_chunk = []

        current_chunk.append(line)

    # Add the last chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if "รายวิชาที่เป็นรหัสวิชาของหลักสูตร" in chunk_text:
            sub_chunks = re.split(split_pattern, chunk_text)
            chunks.extend([sub.strip() for sub in sub_chunks if sub.strip()])
        else:
            chunks.append(chunk_text)

    
    huggingface_ef = HuggingFaceInferenceAPIEmbeddings(
        api_key="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",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    pc = Pinecone(api_key=pinecone_api_key,pool_threads=1
)
    index_list = pc.list_indexes()
    
    # Extract the index names from the index_list
    index_names = [index_info["name"] for index_info in index_list.get("indexes", [])]
    print("Index names are:", index_names)

    if index_names:  # Proceed only if there are indexes to delete
        for index_name in index_names:
            pc.delete_index(index_name)
            print(f"Deleted index: {index_name}")    
    pc.create_index(
        name=indexName,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait until the index is ready
    while not pc.describe_index(indexName).status["ready"]:
        time.sleep(1)
     # Store chunks in MongoDB and Pinecone
    index = pc.Index(indexName)    
    vector_store = PineconeVectorStore(index=index, embedding=huggingface_ef)    
    db10 = client.get_database('totalChunks')  # Replace with your DB name
    collection = db10["chunks"]
    documents = [
        {
            "indexName": indexName,
            "fileName": fileName,
            "chunk_number": i,
            "text": chunk
        }
        for i, chunk in enumerate(chunks, start=1)
    ]

    collection.insert_many(documents)  # Store all chunks in MongoDB
    print(f"Saved {len(chunks)} chunks to MongoDB")

    cursor = collection.find({"indexName": indexName}, batch_size=10)

    index = pc.Index(indexName)    
    print("index is ",index)
    
    texts = ["Tonight, I call on the Senate to: Pass the Freedom to Vote Act.", "ne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.", "One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence."]    
    # vectorstore_from_texts = PineconeVectorStore.from_texts(
    #     texts,
    #     index_name=indexName,
    #     embedding=huggingface_ef,
    #     pinecone_api_key="pcsk_2rixW1_8cyc4WuwQaqjBbKshbJAz1YdoLLZRks3ku4JkQ7E97ij9gSPCLMiprpQrsJ2GyT"
    # )
    # print("PineconeVectorStore is ",PineconeVectorStore)
    # print("vectorstore_from_texts is ",vectorstore_from_texts)
    # Create a PineconeVectorStore instance with API key        
    # Your texts
    texts = ["Hello world", "Pinecone is great", "Vector search is powerful"]
    vector_store = PineconeVectorStore(index=index, embedding=huggingface_ef)
    print("successful vector_from_1txts",vector_store)


    # for batch in cursor:
    #     chunk_text = batch["text"]
    #     print("==")
    #     print(chunk_text)
    #     print("==")
    #     doc = Document(
    #         page_content=chunk_text,
    #         metadata={
    #             "indexName": batch["indexName"],
    #             "fileName": batch["fileName"],
    #             "chunk_number": batch["chunk_number"]
    #         }
    #     )
        

    print(f"All chunks for {indexName} processed successfully")


    # process chunks
    # # Save chunks to MongoDB
    # try:
    #     for i, chunk in enumerate(chunks, start=1):        
    #         doc = Document(
    #             page_content=chunk,
    #             metadata={
    #                 "indexName": indexName,
    #                 "fileName": fileName,
    #                 "chunk_number": i
    #             }
    #         )
    #         print(f"Adding document {i}")  # Debug log
    #         vector_store.add_documents([doc])
    #         print(f"Successfully added document {i}")  # Debug log
    # except Exception as e:
    #     print(f"Error adding documents: {str(e)}")
    #     # Log the full error details
    #     import traceback
    #     print(traceback.format_exc())

# @app.route("/register", methods=['post', 'get'])    
# def index():    
#     message = ''
#     #if method post in index
#     if "email" in session:
#         return redirect(url_for("create_knowledgebase"))
#     if request.method == "POST":
#         user = request.form.get("fullname")
#         email = request.form.get("email")
#         password1 = request.form.get("password1")
#         password2 = request .form.get("password2")
#         #if found in database showcase that it's found 
#         user_found = records.find_one({"name": user})
#         email_found = records.find_one({"email": email})
#         if user_found:
#             message = 'There already is a user by that name'
#             return render_template('index.html', message=message)
#         if email_found:
#             message = 'This email already exists in database'
#             return render_template('index.html', message=message)
#         if password1 != password2:
#             message = 'Passwords should match!'
#             return render_template('index.html', message=message)
#         else:
#             #hash the password and encode it
#             hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
#             #assing them in a dictionary in key value pairs
#             user_input = {'name': user, 'email': email, 'password': hashed}
#             #insert it in the record collection
#             records.insert_one(user_input)
            
#             #find the new created account and its email
#             user_data = records.find_one({"email": email})
#             new_email = user_data['email']
#             #if registered redirect to logged in as the registered user
#             return render_template('create.html', email=new_email)
#     return render_template('index.html')



@app.route("/", methods=["POST", "GET"])
def login():    
    message = 'Please login to your account'    
    if "email" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        #check if email exists in database
        email_found = records.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            #encode the password and check if it matches
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('dashboard'))
            else:
                if "email" in session:
                    return redirect(url_for("dashboard"))
                message = 'Wrong password'
                return render_template('login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('login.html', message=message)
    return render_template('login.html', message=message)

@app.route('/create', methods=["GET", "POST"])
def create_knowledgebase():
    if "email" in session:
        if request.method == "POST":        
            indexName = request.form.get("indexName")        
            courseFile = request.files.get("courseFile")  
            
            if not indexName or not courseFile:
                message = "Please provide both an index name and a file to upload."
                return render_template("create.html", message=message)

            try:
                file_like = io.BytesIO(courseFile.read())

                # Use existing event loop safely                
                chunkByLlama(indexName, file_like, courseFile.filename)

                message = f"Successfully uploaded and processed file for index: {indexName}"
            except Exception as e:
                message = f"Error processing the file: {str(e)}"

            return render_template("create.html", message=message)
        else:
            return render_template("create.html", email=session["email"])
    else:
        return redirect(url_for("login"))
    
@app.route("/logout", methods=["POST", "GET"])

def logout():
    message = 'Please login to your account'    
    if "email" in session:
        session.pop("email", None)
        return render_template("signout.html")
    else:
        return render_template('login.html',message=message)
    
@app.route('/admin-dashboard',methods=["GET", "POST"])
def dashboard():
    if "email" in session:
        if request.method == "POST":
            pc = Pinecone(api_key="pcsk_2rixW1_8cyc4WuwQaqjBbKshbJAz1YdoLLZRks3ku4JkQ7E97ij9gSPCLMiprpQrsJ2GyT")
            index_list = pc.list_indexes()    
            print("Delete button have been trigger")
            index_name = request.form.get("index_name")            
            print("the index name is ",index_name)
            pc.delete_index(index_name)
            return render_template('success.html')
        else:
            pc = Pinecone(api_key="pcsk_2rixW1_8cyc4WuwQaqjBbKshbJAz1YdoLLZRks3ku4JkQ7E97ij9gSPCLMiprpQrsJ2GyT")
            index_list = pc.list_indexes()    
            return render_template('dashboard.html',indexList=index_list)

if __name__ == "__main__":
  app.run(debug=True)
