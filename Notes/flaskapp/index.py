import os
import requests
import uuid 
import time
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
import mlflow

# Set up local MLflow tracking (no authentication needed)
mlflow.set_tracking_uri("file:./mlruns")  # Local directory for tracking

load_dotenv()  # Load environment variables from .env file


app = Flask(__name__)

DATABRICKS_GATEWAY_URL = os.environ.get("DATABRICKS_GATEWAY_URL")  # Set this in your environment
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")  # Set this in your environment

# Store conversation history in memory (for demo purposes)
conversation_history = [
    {"role": "assistant", "content": "Hello! How can I assist you today?"}
]

def chat_response(user_message):
    """Generate a response from the chat model.

    Args:
        user_message (str): The message from the user.

    Returns:
        str: The response from the chat model.
    """
    start = time.time()
    conversation_history.append({"role": "user", "content": user_message})
    payload = {
        "model": "gemini_aigateway",
        "max_tokens": 1024,
        "messages": conversation_history
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DATABRICKS_TOKEN}"
    }
    max_retries = 3
    backoff = 2  # seconds
    for attempt in range(max_retries):
        try:
            resp = requests.post(DATABRICKS_GATEWAY_URL, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            bot_reply = data['choices'][0]['message']['content']
            conversation_history.append({"role": "assistant", "content": bot_reply})
            end = time.time()
            with mlflow.start_run(run_name="Chat Response", run_id="d16dbe3ed056405bb414f77930955027"):
                # mlflow.log_param("user_message_length", len(user_message))
                # mlflow.log_param("response_length", len(bot_reply))
                mlflow.log_metric("response_time_seconds", end - start)
                
            print(f"Response time: {end - start} seconds")
            return bot_reply
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))  # Exponential backoff
            else:
                return "Sorry, the server took too long to respond after several attempts. Please try again later."
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"

# Home page with chat UI
@app.route('/')
def index():
    """Render the chat interface using the templates/index.html file.

    Returns:
        str: The rendered HTML content for the chat interface.
    """
    return render_template('index.html')

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages.

    Returns:
        str: The JSON response containing the chat model's reply.
    """
    data = request.get_json()
    # Generate a unique request ID if not provided
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))      
    print(f"Received message: {data.get('message', '')} with Request ID: {request_id}")

    user_message = data.get('message', '')
    response = chat_response(user_message)
    return jsonify({'response': response, 'request_id': request_id})

if __name__ == '__main__':
    app.run(debug=True)
