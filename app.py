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

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


# connect to LLAMA CLOUD
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-ZnFRL1EnJ5sIhEMLjw3pS75vGMv9gzRzAZr6WMe2BPzzbFoD"
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


def chunkByLlama(file_like, fileName):
    document = LlamaParse(
        result_type="text"
    ).load_data(file_like, extra_info={"file_name": "file_name"})

    document_text = "\n".join([f"- {index} -\n{page.text}" for index, page in enumerate(document)])

    chunk_boundaries = [
        "หมวดที่ 1 ข้อมูลทั่วไป",
        "ชื่อปริญญาและสาขาวิชา",
        "3. หลักสูตรและอาจารย์ผู้สอน",
        "3.1.3 รายวิชา",
        "(2) หมวดวิชาเฉพาะ",
        "แสดงตัวอย่างแผนการศึกษา",
        "รายวิชาที่เป็นรหัสวิชาของหลักสูตร"
    ]

    split_pattern = r'- \d{1,2} -'
    chunks = []
    current_chunk = []

    lines = document_text.split("\n")

    for line in lines:
        if any(boundary in line for boundary in chunk_boundaries) and current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if "รายวิชาที่เป็นรหัสวิชาของหลักสูตร" in chunk_text:
                sub_chunks = re.split(split_pattern, chunk_text)
                chunks.extend([sub.strip() for sub in sub_chunks if sub.strip()])
            else:
                chunks.append(chunk_text)
            current_chunk = []
        current_chunk.append(line)

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

    pc = Pinecone(api_key=pinecone_api_key)
    index_list = pc.list_indexes()
    
    # Extract the index names from the index_list
    index_names = [index_info["name"] for index_info in index_list.get("indexes", [])]
    print("Index names are:", index_names)
    
    # Always use "comsci" as the index name
    index_name_to_use = "comsci"
    
    # Check if index exists
    if index_name_to_use in index_names:
        print(f"Index '{index_name_to_use}' already exists. Skipping creation.")
    else:
        print(f"Creating index '{index_name_to_use}'...")
        pc.create_index(
            name=index_name_to_use,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready (important!)
        
        time.sleep(10)  # Add a delay to allow the index to initialize
    
    # Always use the "comsci" index
    index = pc.Index(index_name_to_use)
    
    # First, check for existing documents with metadata "type":"english" and delete them
    try:
        results = index.query(
            vector=[0] * 384,  # dummy vector
            filter={"type": {"$eq": "course"}},  # Use explicit $eq operator
            top_k=100,
            include_metadata=True
        )
        
        # Extract IDs of documents with "type":"english"
        existing_ids = [match['id'] for match in results['matches']]
        
        if existing_ids:
            print(f"Found {len(existing_ids)} existing documents with type=english: {existing_ids}")
            # Delete these documents
            index.delete(ids=existing_ids)
            print(f"Deleted {len(existing_ids)} documents with type=english")
    except Exception as e:
        print(f"Error when checking/deleting existing documents: {e}")
    
    # Now add the new document
    vector_store = PineconeVectorStore(index=index, embedding=huggingface_ef)    
    for i, chunk in enumerate(chunks, start=1):
        doc_id = "course"+str(i)        
        doc = Document(
            page_content=chunk,
            metadata={
                "type": "course",                                
            }
        )
        vector_store.add_documents([doc], ids=[doc_id])
    print(f"Added document with ID: {doc_id}")        
    time.sleep(2)  # Short delay to ensure document is indexed    
    try:
        results = index.query(
            vector=[0] * 384,  # dummy vector
            filter={"type": {"$eq": "course"}},  # Use explicit $eq operator
            top_k=100,
            include_metadata=True
        )
        
        # Extract IDs
        ids = [match['id'] for match in results['matches']]
        print("Current ids with type=english:", ids)
    except Exception as e:
        print(f"Error querying for documents after addition: {e}")
    print(f"Successfully saved {len(chunks)} records to Pinecone.")

def chunkByLlamaEnglish(file_like, fileName):
    document = LlamaParse(
        result_type="markdown",
        user_prompt="Keep Thai languages",
        system_prompt="Extract the logical flow from this flowchart image and present it in a clear, structured step-by-step format. Ensure that each decision point is distinctly separated and that conditions are clearly linked to their outcomes.",
        premium_mode=True
    ).load_data(file_like, extra_info={"file_name": "file_name"})
    
    text_only = document[0].text_resource.text
    print(text_only)
    
    huggingface_ef = HuggingFaceInferenceAPIEmbeddings(
        api_key="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",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    pc = Pinecone(api_key=pinecone_api_key)
    index_list = pc.list_indexes()
    
    # Extract the index names from the index_list
    index_names = [index_info["name"] for index_info in index_list.get("indexes", [])]
    print("Index names are:", index_names)
    
    # Always use "comsci" as the index name
    index_name_to_use = "comsci"
    
    # Check if index exists
    if index_name_to_use in index_names:
        print(f"Index '{index_name_to_use}' already exists. Skipping creation.")
    else:
        print(f"Creating index '{index_name_to_use}'...")
        pc.create_index(
            name=index_name_to_use,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready (important!)
        
        time.sleep(10)  # Add a delay to allow the index to initialize
    
    # Always use the "comsci" index
    index = pc.Index(index_name_to_use)
    
    # First, check for existing documents with metadata "type":"english" and delete them
    try:
        results = index.query(
            vector=[0] * 384,  # dummy vector
            filter={"type": {"$eq": "english"}},  # Use explicit $eq operator
            top_k=100,
            include_metadata=True
        )
        
        # Extract IDs of documents with "type":"english"
        existing_ids = [match['id'] for match in results['matches']]
        
        if existing_ids:
            print(f"Found {len(existing_ids)} existing documents with type=english: {existing_ids}")
            # Delete these documents
            index.delete(ids=existing_ids)
            print(f"Deleted {len(existing_ids)} documents with type=english")
    except Exception as e:
        print(f"Error when checking/deleting existing documents: {e}")
    
    # Now add the new document
    vector_store = PineconeVectorStore(index=index, embedding=huggingface_ef)
    
    doc = Document(
        page_content=text_only,
        metadata={"type": "english"},
    )
    
    doc_id = "eng1"
    vector_store.add_documents([doc], ids=[doc_id])

    print(f"Added document with ID: {doc_id}")        
    time.sleep(2)  # Short delay to ensure document is indexed    
    try:
        results = index.query(
            vector=[0] * 384,  # dummy vector
            filter={"type": {"$eq": "english"}},  # Use explicit $eq operator
            top_k=100,
            include_metadata=True
        )
        
        # Extract IDs
        ids = [match['id'] for match in results['matches']]
        print("Current ids with type=english:", ids)
    except Exception as e:
        print(f"Error querying for documents after addition: {e}")
    
    return doc_id

@app.route("/register", methods=['post', 'get'])    
def index():    
    message = ''
    #if method post in index
    if "email" in session:
        return redirect(url_for("create_knowledgebase"))
    if request.method == "POST":
        user = request.form.get("fullname")
        email = request.form.get("email")
        password1 = request.form.get("password1")
        password2 = request .form.get("password2")
        #if found in database showcase that it's found 
        user_found = records.find_one({"name": user})
        email_found = records.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('index.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('index.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('index.html', message=message)
        else:
            #hash the password and encode it
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            #assing them in a dictionary in key value pairs
            user_input = {'name': user, 'email': email, 'password': hashed}
            #insert it in the record collection
            records.insert_one(user_input)
            
            #find the new created account and its email
            user_data = records.find_one({"email": email})
            new_email = user_data['email']
            #if registered redirect to logged in as the registered user
            return render_template('create.html', email=new_email)
    return render_template('index.html')



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
            courseFile = request.files.get("courseFile")  
            try:
                file_like = io.BytesIO(courseFile.read())
                # Use existing event loop safely                
                chunkByLlama(file_like, courseFile.filename)
                message = f"Successfully uploaded and processed file for index"
            except Exception as e:
                message = f"Error processing the file: {str(e)}"

            return render_template("create.html", message=message)
        else:
            return render_template("create.html", email=session["email"])
    else:
        return redirect(url_for("login"))
    
@app.route('/create-english', methods=["GET", "POST"])
def create_english():
    if "email" in session:
        if request.method == "POST":        
            # indexName = request.form.get("indexName")        
            courseFile = request.files.get("courseFile")  
            
            # if not indexName or not courseFile:
            #     message = "Please provide both an index name and a file to upload."
            #     return render_template("create.html", message=message)

            try:
                file_like = io.BytesIO(courseFile.read())
                print("file_like",file_like)

                # Use existing event loop safely                
                chunkByLlamaEnglish(file_like, courseFile.filename)

                message = f"Successfully uploaded and processed file for index: "
            except Exception as e:
                message = f"Error processing the file: {str(e)}"

            return render_template("create-english.html", message=message)
        else:
            return render_template("create-english.html", email=session["email"])
    else:
        return redirect(url_for("login"))
    
@app.route('/create-ged-ed', methods=["GET", "POST"])    
def create_gen():
    if "email" in session:
        if request.method == "POST":                    
            import gspread
            from google.oauth2.service_account import Credentials
            service_account_info = {
            "type": "service_account",
            "project_id": "final-rag",
            "private_key_id": "593d6aaa7e5bb6cae44439c8954badd6b30bad8d",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQClXRNvgJZhpPgK\nwg8SuS/bSWijlBj+U/VDuGWIz6V+go7mPnGVRq+Md0kHFMXzM+gJ/2ySTl95wMph\nvfVbrRNNKMryBxrtBq4LfKrygx0NTGaE/C7e4hYSNSsRyDab+JsGw/CJIRgC3Nxu\nPIwP4JuPKK3A+K/OKpyItqeZKShXEDzkWyfqFfeoIdBLhTKhbGKYicKtlGGtE9Hq\nCRiH00/ou9wPJEsSqrk6k0HI7JodZd5X1GVw7DbCjp676hLnRpHPwaN80l+6BwCO\nvPJwcbUCz/ihP8sGWPsX9mZBy5zdEiNbZKSJiaK0SVm3iGWFbCb5DYzW3wZbZPTG\ny2T96Js7AgMBAAECggEAGUMq4PHoB2rIafxTiSy5XurMDZFmcBQrd/kHqervAXC4\nm/wWJhPyZacjhO1rgEgBvuVClOdcNqF5SY1XmnXKaRM+TdNADc3jcOXOx9W37nwp\nfU95aZtoe8ebmM/ZZ+KG7HWqnYhsvqM3GbAuRY6utSQlx0E2umxQFaKx3/glrYcS\nfUsqWQysRoALn6ZQkF14Mu7j4PU6cBeTEMhCUDgcUKbGYJqh3VTf6WLNbMrDbkho\n7kfgOeSC8bqHr93lhTxrIJ9d3GtnbTD4pPNpbB5RUKbj+XKRL3CbAppU+WvG9VF/\neVksWJySoOGhYi/J9ztu/QZ2upJRjsXK18h8o0ZBIQKBgQDddSphvCa4w4D9Cs5h\npN4NoQOdes4j3XNaIOAvkb0G4nx0WrJQg0E8cWkMXh9EyA9wfYU8YDjTzgbuZuwT\nyGGRj6yS9NnMgHRSATivn0GVjfDK0+gcToPXGRbDLJWVi9Astp2En9PxwmQfgHh6\nctc+xltngMJEjYWf4JrqUftcRQKBgQC/KBDFQOGBs4blDox31aN9JsylDH6Pfy95\nGzQU03SSPEX5xc55/6R4OlEAjNsIZONI9tLuhHWxI9jCK1Modn8fn3F1sFaSZDZh\nfSad2x2S9DOk2B5xMrlcXuCrrjLZKY/7EjKtURarYn7+l5T6IeMx03BJG2De0rIv\noUWJnDRRfwKBgQDIj6j/dJ/46w4xnQzF/8Mewrj8cVCpyJAEiwud6TYxOwMNeWpO\nYmC9ddR2X/OfnjPlY7g7ssUkhU1fsZSSYgKDCoR3Xwq1G4y9C+AjpW6HHFJ7zqhC\nopTiRBWKUyFxm3rAU+6aQwl2xN9abEYwVzs63fe/6CuIXEctQQPrvK2RpQKBgQCR\nyH+Jv+J7pSvicscD+UV3A+kckrvOukO9S9bbbyy+/gKr64R9nE6VdnwiPEorS63f\nDoZta03Kq7j61EnWWRC4UEQaakKL4KtsjCKwTtRuJ5lfRYdp8zJUVPNpWy/iWIU7\nCHTnoyjzyelqRrZSURfQ/xzqVFv7c5p7IrZCrYNlBwKBgQDaMryRIqGk0s1wMMbO\nSVI4zatI/31s0aPJd9bY8KF1l01xcthFXcEWNwjyA9MTZeByBEoAfMXdl3Ofpl1x\nmCZMmi+OpnZrOTCAIfeGa5gCVibgKUm7mZqLwyMHUlzjVR/TnaDrRPWXn1Iyf6fu\n5CT+I8Tn1Ch5yMBqXlsw2tY6pw==\n-----END PRIVATE KEY-----\n",
            "client_email": "python-api@final-rag.iam.gserviceaccount.com",
            "client_id": "118371841608472169473",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/python-api%40final-rag.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
            }
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets"
            ]
            creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
            client = gspread.authorize(creds)

            sheet_id = "15Etw5nYr_XW3nlH7HrV634PsHyEcRErSF4EuzNdLTRw"
            sheet = client.open_by_key(sheet_id)

            all_rows = sheet.sheet1.get_all_values()

            # for row in all_rows:
            #     print(row)      
            chunk_size = 5  # Number of subjects per chunk
            subjects = all_rows[1:]  # Exclude the header row

            chunks = [subjects[i:i + chunk_size] for i in range(0, len(subjects), chunk_size)]

            # Example output of one chunk
            # Example output of one chunk
            chunked_data = {}
            # ===
            huggingface_ef = HuggingFaceInferenceAPIEmbeddings(
            api_key="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",
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            pc = Pinecone(api_key=pinecone_api_key)
            index_list = pc.list_indexes()
    
            # Extract the index names from the index_list
            index_names = [index_info["name"] for index_info in index_list.get("indexes", [])]
            print("Index names are:", index_names)
    
            # Always use "comsci" as the index name
            index_name_to_use = "comsci"    
            index = pc.Index(index_name_to_use)

            if index_name_to_use in index_names:
                print(f"Index '{index_name_to_use}' already exists. Skipping creation.")
            else:
                print(f"Creating index '{index_name_to_use}'...")
                pc.create_index(
                    name=index_name_to_use,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                # Wait for index to be ready (important!)
                
                time.sleep(10)  # Add a delay to allow the index to initialize
    
                # Always use the "comsci" index
                index = pc.Index(index_name_to_use)
    
                # First, check for existing documents with metadata "type":"english" and delete them
                try:
                    results = index.query(
                        vector=[0] * 384,  # dummy vector
                        filter={"type": {"$eq": "ged-ed"}},  # Use explicit $eq operator
                        top_k=100,
                        include_metadata=True
                    )
                    
                    # Extract IDs of documents with "type":"english"
                    existing_ids = [match['id'] for match in results['matches']]
                    
                    if existing_ids:
                        print(f"Found {len(existing_ids)} existing documents with type=ged-ed: {existing_ids}")
                        # Delete these documents
                        index.delete(ids=existing_ids)
                        print(f"Deleted {len(existing_ids)} documents with type=ged-ed")
                except Exception as e:
                    print(f"Error when checking/deleting existing documents: {e}")
    
                # Now add the new document
            
            vector_store = PineconeVectorStore(index=index, embedding=huggingface_ef)
            # Loop through all chunks and process them
            for idx, chunk in enumerate(chunks):
                chunk_name = f"chunk_{idx + 1}"
                chunked_data[chunk_name] = "\n".join([
                    f"รหัสวิชา: {subject[0]}\n"
                    f"ชื่อวิชา: {subject[1]}\n"
                    f"หน่วยกิต: {subject[2]}\n"
                    f"คณะ: {subject[3]}\n"
                    f"หมวดหมู่: {subject[4]}\n"
                    f"คำอธิบายรายวิชา: {subject[5]}\n"
                    f"{'-'*40}"
                    for subject in chunk
                ])

                # Create document dynamically
                doc = Document(
                    page_content=chunked_data[chunk_name],
                    metadata={"type": "ged-ed"}
                )

                doc_id = f"ged-ed{idx + 1}"  # Unique ID for each chunk
                vector_store.add_documents([doc], ids=[doc_id])
                print(f"Added document with ID: {doc_id}")                
            time.sleep(2)  # Short delay to ensure document is indexed    
            try:
                results = index.query(
                vector=[0] * 384,  # dummy vector
                filter={"type": {"$eq": "ged-ed"}},  # Use explicit $eq operator
                top_k=100,
                include_metadata=True )
                    # Extract IDs
                ids = [match['id'] for match in results['matches']]
                print("Current ids with type=ged-ed:", ids)
            except Exception as e:
                print(f"Error querying for documents after addition: {e}") 
            # ===            
            return render_template("create-gen-ed.html")
        else:
            return render_template("create-gen-ed.html", email=session["email"])
    else:
        return redirect(url_for("login"))
    
@app.route('/create-policy', methods=["GET", "POST"])    
def create_policy():
    if "email" in session:
        if request.method == "POST":                    
            import gspread
            from google.oauth2.service_account import Credentials
            service_account_info = {
            "type": "service_account",
            "project_id": "final-rag",
            "private_key_id": "593d6aaa7e5bb6cae44439c8954badd6b30bad8d",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQClXRNvgJZhpPgK\nwg8SuS/bSWijlBj+U/VDuGWIz6V+go7mPnGVRq+Md0kHFMXzM+gJ/2ySTl95wMph\nvfVbrRNNKMryBxrtBq4LfKrygx0NTGaE/C7e4hYSNSsRyDab+JsGw/CJIRgC3Nxu\nPIwP4JuPKK3A+K/OKpyItqeZKShXEDzkWyfqFfeoIdBLhTKhbGKYicKtlGGtE9Hq\nCRiH00/ou9wPJEsSqrk6k0HI7JodZd5X1GVw7DbCjp676hLnRpHPwaN80l+6BwCO\nvPJwcbUCz/ihP8sGWPsX9mZBy5zdEiNbZKSJiaK0SVm3iGWFbCb5DYzW3wZbZPTG\ny2T96Js7AgMBAAECggEAGUMq4PHoB2rIafxTiSy5XurMDZFmcBQrd/kHqervAXC4\nm/wWJhPyZacjhO1rgEgBvuVClOdcNqF5SY1XmnXKaRM+TdNADc3jcOXOx9W37nwp\nfU95aZtoe8ebmM/ZZ+KG7HWqnYhsvqM3GbAuRY6utSQlx0E2umxQFaKx3/glrYcS\nfUsqWQysRoALn6ZQkF14Mu7j4PU6cBeTEMhCUDgcUKbGYJqh3VTf6WLNbMrDbkho\n7kfgOeSC8bqHr93lhTxrIJ9d3GtnbTD4pPNpbB5RUKbj+XKRL3CbAppU+WvG9VF/\neVksWJySoOGhYi/J9ztu/QZ2upJRjsXK18h8o0ZBIQKBgQDddSphvCa4w4D9Cs5h\npN4NoQOdes4j3XNaIOAvkb0G4nx0WrJQg0E8cWkMXh9EyA9wfYU8YDjTzgbuZuwT\nyGGRj6yS9NnMgHRSATivn0GVjfDK0+gcToPXGRbDLJWVi9Astp2En9PxwmQfgHh6\nctc+xltngMJEjYWf4JrqUftcRQKBgQC/KBDFQOGBs4blDox31aN9JsylDH6Pfy95\nGzQU03SSPEX5xc55/6R4OlEAjNsIZONI9tLuhHWxI9jCK1Modn8fn3F1sFaSZDZh\nfSad2x2S9DOk2B5xMrlcXuCrrjLZKY/7EjKtURarYn7+l5T6IeMx03BJG2De0rIv\noUWJnDRRfwKBgQDIj6j/dJ/46w4xnQzF/8Mewrj8cVCpyJAEiwud6TYxOwMNeWpO\nYmC9ddR2X/OfnjPlY7g7ssUkhU1fsZSSYgKDCoR3Xwq1G4y9C+AjpW6HHFJ7zqhC\nopTiRBWKUyFxm3rAU+6aQwl2xN9abEYwVzs63fe/6CuIXEctQQPrvK2RpQKBgQCR\nyH+Jv+J7pSvicscD+UV3A+kckrvOukO9S9bbbyy+/gKr64R9nE6VdnwiPEorS63f\nDoZta03Kq7j61EnWWRC4UEQaakKL4KtsjCKwTtRuJ5lfRYdp8zJUVPNpWy/iWIU7\nCHTnoyjzyelqRrZSURfQ/xzqVFv7c5p7IrZCrYNlBwKBgQDaMryRIqGk0s1wMMbO\nSVI4zatI/31s0aPJd9bY8KF1l01xcthFXcEWNwjyA9MTZeByBEoAfMXdl3Ofpl1x\nmCZMmi+OpnZrOTCAIfeGa5gCVibgKUm7mZqLwyMHUlzjVR/TnaDrRPWXn1Iyf6fu\n5CT+I8Tn1Ch5yMBqXlsw2tY6pw==\n-----END PRIVATE KEY-----\n",
            "client_email": "python-api@final-rag.iam.gserviceaccount.com",
            "client_id": "118371841608472169473",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/python-api%40final-rag.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
            }
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets"
            ]
            creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
            client = gspread.authorize(creds)

            sheet_id = "1_LGcK9OX3ZmIAKLpyfIgwB6sAyAEyUDNYRT6qwvWV0M"
            sheet = client.open_by_key(sheet_id)

            all_rows = sheet.sheet1.get_all_values()                   
            huggingface_ef = HuggingFaceInferenceAPIEmbeddings(
            api_key="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",
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            pc = Pinecone(api_key=pinecone_api_key)
            index_list = pc.list_indexes()
    
            # Extract the index names from the index_list
            index_names = [index_info["name"] for index_info in index_list.get("indexes", [])]
            print("Index names are:", index_names)
    
            # Always use "comsci" as the index name
            index_name_to_use = "comsci"    
            index = pc.Index(index_name_to_use)

            if index_name_to_use in index_names:
                print(f"Index '{index_name_to_use}' already exists. Skipping creation.")
            else:
                print(f"Creating index '{index_name_to_use}'...")
                pc.create_index(
                    name=index_name_to_use,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                # Wait for index to be ready (important!)
                
                time.sleep(10)  # Add a delay to allow the index to initialize
                
                # Always use the "comsci" index
                index = pc.Index(index_name_to_use)
    
                # First, check for existing documents with metadata "type":"english" and delete them
                try:
                    results = index.query(
                        vector=[0] * 384,  # dummy vector
                        filter={"type": {"$eq": "policy"}},  # Use explicit $eq operator
                        top_k=100,
                        include_metadata=True
                    )
                    
                    # Extract IDs of documents with "type":"english"
                    existing_ids = [match['id'] for match in results['matches']]
                    print("existing_ids : ",existing_ids)
                    
                    if existing_ids:
                        print(f"Found {len(existing_ids)} existing documents with type=policy: {existing_ids}")
                        # Delete these documents
                        index.delete(ids=existing_ids)
                        print(f"Deleted {len(existing_ids)} documents with type=policy")
                except Exception as e:
                    print(f"Error when checking/deleting existing documents: {e}")
    
                # Now add the new document
            
            vector_store = PineconeVectorStore(index=index, embedding=huggingface_ef)
            idx = 1  # Start from 1

            for row in all_rows:
                 formatted_text = "\n".join(row).strip()  # Ensure no leading/trailing spaces                  
                 print("formatted_text : ",formatted_text)                
                 doc = Document(
                    page_content=formatted_text,
                    metadata={"type": "policy"}
                 )                            
                 doc_id = f"policy-{idx}"  # Use idx for a unique ID
                 vector_store.add_documents([doc], ids=[doc_id])                
                 print(f"Added document with ID: {doc_id}")     
                 idx += 1  # Increment index after each iteration                
            try:
                results = index.query(
                vector=[0] * 384,  # dummy vector
                filter={"type": {"$eq": "policy"}},  # Use explicit $eq operator
                top_k=100,
                include_metadata=True )            
                ids = [match['id'] for match in results['matches']]
                print("Current ids with type=policy:", ids)
            except Exception as e:
                print(f"Error querying for documents after addition: {e}")             
            return render_template("create-policy.html")
        else:
            return render_template("create-policy.html", email=session["email"])
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
