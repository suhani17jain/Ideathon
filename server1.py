from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import openai
import dropbox
import requests
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import json
import re
import os
from flask import Flask, send_from_directory
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/images/<path:filename>')
def serve_image(filename):
    print("filename", filename)
    return send_from_directory('images', filename)

# Read the OpenAI API key
openai.api_key = open("OpenAI_API_Key.txt", "r").read().strip()

# Initialize ChromaDB PersistentClient
client = PersistentClient()

# Set up the embedding function using OpenAI embedding model
model = "text-embedding-ada-002"
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model)

# Initialize the main collection in ChromaDB
photoshop_collection = client.get_or_create_collection(name='RAG_on_photoshop_data', embedding_function=embedding_function)

# Initialize CrossEncoder for re-ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Dropbox setup
DROPBOX_ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Adobe setup
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

def get_access_token(client_id, client_secret):
    params = {
        'client_secret': client_secret,
        'grant_type': 'client_credentials',
        'scope': 'openid,AdobeID,read_organizations'
    }
    response = requests.post(
        f'https://ims-na1.adobelogin.com/ims/token/v2?client_id={client_id}', 
        data=params
    )
    return response.json().get('access_token')

def query_photoshop_action(query):
    results = photoshop_collection.query(query_texts=[query], n_results=1)
    top_result = results['metadatas'][0][0]  # Taking the top result's metadata
    return top_result['json']

def remove_json_comments(json_string):
    """Remove comments from a JSON string."""
    # Remove single-line comments
    json_string_without_single_line_comments = re.sub(r'//.*', '', json_string)
    # Remove multi-line comments
    json_string_without_comments = re.sub(r'/\*[\s\S]*?\*/', '', json_string_without_single_line_comments)
    return json_string_without_comments.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    file_name = file.filename
    dbx.files_upload(file.read(), f'/{file_name}', mode=dropbox.files.WriteMode.overwrite)
    return jsonify({'fileName': file_name})

@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.json
    file_name = data.get('fileName')
    prompt = data.get('prompt')

    if not file_name or not prompt:
        return jsonify({'error': 'Missing fileName or prompt'}), 400
    output_file_path = f'/modified_{file_name}'
    # Get Dropbox links
    input_link = dbx.files_get_temporary_link(f'/{file_name}').link
    output_link = dbx.files_get_temporary_upload_link(commit_info=dropbox.files.CommitInfo(path=output_file_path, mode=dropbox.files.WriteMode.overwrite))

    # Query the Photoshop action
    action_json_str = query_photoshop_action(prompt)
    print("Raw Action JSON String:", action_json_str)
    
    # Remove comments from the action JSON string
    action_json_str = remove_json_comments(action_json_str)
    print("Cleaned Action JSON String:", action_json_str)

    try:
        action_json_array = json.loads(action_json_str)
        
        # Wrap the array in an object if required by Adobe API
        # action_json = {"actions": action_json_array}
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return jsonify({'error': 'Invalid JSON data'}), 400

    # Get Adobe access token
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    print(type(action_json_array), "ACTIONJSON")
    print(access_token, "ACCESSTOKEN")
    # Prepare data for Adobe API
    data = {
        "inputs": [{"storage": "dropbox", "href": input_link}],
        "options": { "actionJSON":  action_json_array},
        "outputs": [{"storage": "dropbox", "type": "image/vnd.adobe.photoshop", "href": output_link.link}]
    }

    # Call Adobe API
    response = requests.post(
        'https://image.adobe.io/pie/psdService/actionJSON',
        headers={
            'Authorization': f'Bearer {access_token}',
            'x-api-key': CLIENT_ID
        },
        json=data
    )
    result = response.json()
    print(result)
    status = "running"
    # Polling the status
    while status == "running":
        job_response = requests.get(result['_links']['self']['href'], headers={
            'Authorization': f'Bearer {access_token}',
            'x-api-key': CLIENT_ID
        })
        job_result = job_response.json()
        # print(job_result, "JOBRESULT")
        status = job_result["outputs"][0]['status']
        import time
        time.sleep(1)
    print(job_result, "Jobresult")
    # print(job_result["outputs"][0]["_links"]["renditions"][0]["href"])
    # modified_image_url = job_result["outputs"][0]["_links"]["renditions"][0]["href"]

    try:
        shared_link_metadata = dbx.sharing_create_shared_link_with_settings(output_file_path)
    except Exception as e:
        # import pdb;pdb.set_trace()
        shared_link_metadata = e.error.get_shared_link_already_exists().get_metadata()

    print(shared_link_metadata, type(shared_link_metadata))
    common_link = shared_link_metadata.url
    direct_link = common_link.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("dl=0", "dl=1")

    def download_image(url, file_path):
        try:
            # Send a GET request to the URL
            response = requests.get(url, allow_redirects=True)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Write the content of the response (image) to a file
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                    return f"/{file_path}"
                print(f"Image downloaded successfully and saved to {file_path}")
            else:
                print(f"Failed to download image. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")
    import os
    os.makedirs("images", exist_ok=True)
    modified_img_path = f"images/modified_{file_name}"
    hack_img_path = f"images/modified_tricked_{file_name}"
    local_url = download_image(direct_link, modified_img_path)
    image = Image.open(modified_img_path)
    image.save(hack_img_path)
    return jsonify({'modifiedImageUrl': "/"+hack_img_path})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
