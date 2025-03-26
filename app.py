import os
import re
import io
import nest_asyncio



from flask import Flask, render_template, request, url_for, redirect, session
from pymongo import MongoClient
from llama_parse import LlamaParse
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import gspread
from google.oauth2.service_account import Credentials            
import bcrypt
import time
#set app as a Flask instance 
app = Flask(__name__)
#encryption relies on secret keys so they could be run
app.secret_key = "testing"
# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# connect to LLAMA CLOUD
os.environ["LLAMA_CLOUD_API_KEY"] = os.environ.get("LLAMA_CLOUD_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY")

pinecone_api_key = os.environ["PINECONE_API_KEY"]
huggingface_api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]
connection_string = f"""mongodb+srv://admin555:{os.environ.get("MONGODB_PASSWORD")}@rag.8zbcn.mongodb.net/?retryWrites=true&w=majority&appName=RAG"""
#connect to your Mongo DB database
client = MongoClient(connection_string)
db = client.get_database('total_records')
records = db.register    

def chunkByLlama(file_like):
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
        api_key=huggingface_api_key,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    pc = Pinecone(api_key=pinecone_api_key)
    index_list = pc.list_indexes()
    
    # Extract the index names from the index_list
    index_names = [index_info["name"] for index_info in index_list.get("indexes", [])]    
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
            top_k=50,
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
            top_k=50,
            include_metadata=True
        )
        
        # Extract IDs
        ids = [match['id'] for match in results['matches']]
        print("Current ids :", ids)
    except Exception as e:
        print(f"Error querying for documents after addition: {e}")
    print(f"Successfully saved {len(chunks)} records to Pinecone.")

def chunkByLlamaEnglish(file_like):
    document = LlamaParse(
        result_type="markdown",
        user_prompt="Keep Thai languages",
        system_prompt="Extract the logical flow from this flowchart image and present it in a clear, structured step-by-step format. Ensure that each decision point is distinctly separated and that conditions are clearly linked to their outcomes.",        
        premium_mode=True
    ).load_data(file_like, extra_info={"file_name": "file_name"})
    
    text_only = document[0].text_resource.text    
    huggingface_ef = HuggingFaceInferenceAPIEmbeddings(
        api_key=huggingface_api_key,
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
    
    # Always use the "comsci" index
    index = pc.Index(index_name_to_use)    
    # First, check for existing documents with metadata "type":"english" and delete them
    try:
        results = index.query(            
            filter={"type": {"$eq": "english"}},  # Use explicit $eq operator
            top_k=1,
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
            filter={"type": {"$eq": "english"}},  # Use explicit $eq operator
            top_k=1,
            include_metadata=True
        )
        
        # Extract IDs
        ids = [match['id'] for match in results['matches']]
        print("Current ids :", ids)
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
            courseFile = request.files.get("courseFile")  
            try:
                file_like = io.BytesIO(courseFile.read())

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
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets"
            ]
            credentials_json = {
            "type": os.getenv("TYPE"),
            "project_id": os.getenv("PROJECT_ID"),
            "private_key_id": os.getenv("PRIVATE_KEY_ID"),
            "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("CLIENT_EMAIL"),
            "client_id": os.getenv("CLIENT_ID"),
            "auth_uri": os.getenv("AUTH_URI"),
            "token_uri": os.getenv("TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
            "universe_domain": os.getenv("UNIVERSE_DOMAIN")
        }
            creds = Credentials.from_service_account_info(credentials_json, scopes=scopes)            
            client = gspread.authorize(creds)

            sheet_id = "15Etw5nYr_XW3nlH7HrV634PsHyEcRErSF4EuzNdLTRw"
            sheet = client.open_by_key(sheet_id)
            all_rows = sheet.sheet1.get_all_values()            
            chunk_size = 5  # Number of subjects per chunk
            subjects = all_rows[1:]  # Exclude the header row
            chunks = [subjects[i:i + chunk_size] for i in range(0, len(subjects), chunk_size)]
            chunked_data = {}            
            huggingface_ef = HuggingFaceInferenceAPIEmbeddings(
            api_key=huggingface_api_key,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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
                time.sleep(10)    
                # Always use the "comsci" index
                index = pc.Index(index_name_to_use)
                # First, check for existing documents with metadata "type":"english" and delete them
                try:
                    results = index.query(
                        vector=[0] * 384,  # dummy vector
                        filter={"type": {"$eq": "ged-ed"}},  # Use explicit $eq operator
                        top_k=60,
                        include_metadata=True
                    )                    
                    # Extract IDs of documents with "type":"english"
                    existing_ids = [match['id'] for match in results['matches']]
                    
                    if existing_ids:
                        print(f"Found {len(existing_ids)} existing documents with : {existing_ids}")
                        # Delete these documents
                        index.delete(ids=existing_ids)
                        print(f"Deleted {len(existing_ids)}")
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
                top_k=60,
                include_metadata=True )
                    # Extract IDs
                ids = [match['id'] for match in results['matches']]
                print("Current ids:", ids)
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
            
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets"
            ]
            credentials_json = {
            "type": os.getenv("TYPE"),
            "project_id": os.getenv("PROJECT_ID"),
            "private_key_id": os.getenv("PRIVATE_KEY_ID"),
            "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("CLIENT_EMAIL"),
            "client_id": os.getenv("CLIENT_ID"),
            "auth_uri": os.getenv("AUTH_URI"),
            "token_uri": os.getenv("TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
            "universe_domain": os.getenv("UNIVERSE_DOMAIN")
        }
            creds = Credentials.from_service_account_info(credentials_json, scopes=scopes)
            client = gspread.authorize(creds)

            # Open Google Sheet
            sheet_id = "1_LGcK9OX3ZmIAKLpyfIgwB6sAyAEyUDNYRT6qwvWV0M"
            sheet = client.open_by_key(sheet_id)        
            all_rows = sheet.sheet1.get_all_values()                   
            huggingface_ef = HuggingFaceInferenceAPIEmbeddings(
            api_key=huggingface_api_key,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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
                # Always use the "comsci" index
                index = pc.Index(index_name_to_use)                    
                try:
                    results = index.query(
                        vector=[0] * 384,  # dummy vector
                        filter={"type": {"$eq": "policy"}},  # Use explicit $eq operator
                        top_k=50,
                        include_metadata=True
                    )                                        
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
                top_k=50,
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
            pc = Pinecone(api_key=pinecone_api_key)
            index_list = pc.list_indexes()    
            print("Delete button have been trigger")
            index_name = request.form.get("index_name")                        
            pc.delete_index(index_name)
            return render_template('success.html')
        else:
            pc = Pinecone(api_key=pinecone_api_key)
            index_list = pc.list_indexes()    
            return render_template('dashboard.html',indexList=index_list)

if __name__ == "__main__":
  app.run(debug=True)
