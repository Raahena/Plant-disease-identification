import json
import time
import os
from pathlib import Path
import google.generativeai as genai
import argparse

# Configure your Gemini API credentials
GOOGLE_API_KEY = "AIzaSyDnExSB0B7gm39mMYC9gpEL-MdtMweP6dE"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Path to shared data files
SHARED_DIR = Path("../Plant_Disease_Dataset/shared")
QUEUE_FILE = SHARED_DIR / "disease_queue.json"
RESULTS_FILE = SHARED_DIR / "recommendations.json"
CHAT_QUEUE_FILE = SHARED_DIR / "chat_queue.json"
CHAT_RESULTS_FILE = SHARED_DIR / "chat_responses.json"

def initialize_files():
    """Ensure shared directory and files exist"""
    SHARED_DIR.mkdir(exist_ok=True)
    
    if not QUEUE_FILE.exists():
        with open(QUEUE_FILE, "w") as f:
            json.dump({"requests": [], "processed": []}, f)
            
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            json.dump({}, f)
            
    if not CHAT_QUEUE_FILE.exists():
        with open(CHAT_QUEUE_FILE, "w") as f:
            json.dump({"requests": [], "processed": []}, f)
            
    if not CHAT_RESULTS_FILE.exists():
        with open(CHAT_RESULTS_FILE, "w") as f:
            json.dump({}, f)

def get_recommendation(disease_name):
    """Use Gemini to generate a recommendation for the given plant disease"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    Provide a precise recommendation for treating {disease_name} in plants.
    Include:
    1. Brief description of the disease
    2. Symptoms identification
    3. Treatment options (organic and chemical)
    4. Prevention strategies 
    5. Expected recovery time in 2 lines
    give everything in bullet points
    Format the response in markdown.
    """
    
    response = model.generate_content(prompt)
    return response.text

def handle_chat_query(query, context=None):
    """Process a chat query using Gemini"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Craft the prompt based on whether there's context or not
    if context:
        prompt = f"""
        You are a helpful plant care assistant. The user previously identified a plant with {context}.
        
        Please respond to the following question in a helpful, concise manner:
        
        User question: {query}
        
        Your answer should be informative, friendly, and focused on plant care. If you don't know the answer,
        acknowledge that and suggest the user consult with a plant expert. Keep your response under 200 words.
        """
    else:
        prompt = f"""
        You are a helpful plant care assistant. 
        
        Please respond to the following question in a helpful, concise manner:
        
        User question: {query}
        
        Your answer should be informative, friendly, and focused on plant care. If you don't know the answer,
        acknowledge that and suggest the user consult with a plant expert. Keep your response under 200 words.
        """
    
    response = model.generate_content(prompt)
    return response.text

def service_loop():
    """Main service loop checking for new requests"""
    print("Starting Gemini recommendation service...")
    initialize_files()
    
    while True:
        # Process disease recommendation requests
        try:
            with open(QUEUE_FILE, "r") as f:
                queue_data = json.load(f)
                
            # Get unprocessed requests
            new_requests = []
            for req in queue_data["requests"]:
                # Convert processed items to just IDs for easier comparison
                processed_ids = [p["id"] for p in queue_data.get("processed", [])]
                if req["id"] not in processed_ids:
                    new_requests.append(req)
            
            # Process new disease recommendation requests
            for request in new_requests:
                request_id = request["id"]
                disease_name = request["disease"]
                print(f"Processing recommendation for: {disease_name}")
                
                try:
                    recommendation = get_recommendation(disease_name)
                    
                    # Load existing results
                    try:
                        with open(RESULTS_FILE, "r") as f:
                            results = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        results = {}
                    
                    # Add new result
                    results[request_id] = {
                        "disease": disease_name,
                        "recommendation": recommendation,
                        "timestamp": time.time()
                    }
                    
                    # Save updated results
                    with open(RESULTS_FILE, "w") as f:
                        json.dump(results, f)
                    
                    # Mark as processed
                    queue_data["processed"].append(request)
                    with open(QUEUE_FILE, "w") as f:
                        json.dump(queue_data, f)
                        
                    print(f"Completed recommendation for: {disease_name}")
                    
                except Exception as e:
                    print(f"Error processing {disease_name}: {str(e)}")
        except Exception as e:
            print(f"Error processing disease queue: {str(e)}")
        
        # Process chat requests
        try:
            with open(CHAT_QUEUE_FILE, "r") as f:
                chat_queue_data = json.load(f)
            
            # Ensure proper structure
            if "requests" not in chat_queue_data:
                chat_queue_data = {"requests": [], "processed": []}
                with open(CHAT_QUEUE_FILE, "w") as f:
                    json.dump(chat_queue_data, f)
                continue
                
            # Get unprocessed chat requests
            new_chat_requests = []
            processed_ids = [p.get("id", "") for p in chat_queue_data.get("processed", [])]
            
            for req in chat_queue_data.get("requests", []):
                if req.get("id") and req.get("id") not in processed_ids:
                    # Make sure it has the required fields
                    if "query" in req:
                        new_chat_requests.append(req)
            
            # Process new chat requests
            for request in new_chat_requests:
                request_id = request["id"]
                query = request["query"]
                context = request.get("context")
                print(f"Processing chat request: {query[:30]}...")
                
                try:
                    chat_response = handle_chat_query(query, context)
                    
                    # Load existing results
                    try:
                        with open(CHAT_RESULTS_FILE, "r") as f:
                            results = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        results = {}
                    
                    # Add new result
                    results[request_id] = {
                        "query": query,
                        "response": chat_response,
                        "timestamp": time.time()
                    }
                    
                    # Save updated results
                    with open(CHAT_RESULTS_FILE, "w") as f:
                        json.dump(results, f)
                    
                    # Mark as processed
                    if "processed" not in chat_queue_data:
                        chat_queue_data["processed"] = []
                    chat_queue_data["processed"].append(request)
                    with open(CHAT_QUEUE_FILE, "w") as f:
                        json.dump(chat_queue_data, f)
                        
                    print(f"Completed chat response for: {query[:30]}...")
                    
                except Exception as e:
                    print(f"Error processing chat request: {str(e)}")
        except Exception as e:
            print(f"Error processing chat queue: {str(e)}")
            # Reset the chat queue if there's an error
            try:
                with open(CHAT_QUEUE_FILE, "w") as f:
                    json.dump({"requests": [], "processed": []}, f)
            except:
                pass
        
        # Sleep before checking again
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gemini Service')
    parser.add_argument('--test-disease', type=str, help='Test a specific disease name')
    parser.add_argument('--test-chat', type=str, help='Test a chat query')
    parser.add_argument('--init', action='store_true', help='Initialize chat queue files')
    
    args = parser.parse_args()
    
    if args.init:
        print("Initializing chat queue files...")
        initialize_files()
        print("Done.")
    elif args.test_disease:
        # Test mode - just generate one disease recommendation
        print(f"Testing recommendation for: {args.test_disease}")
        recommendation = get_recommendation(args.test_disease)
        print("\n===== RECOMMENDATION =====\n")
        print(recommendation)
    elif args.test_chat:
        # Test mode - just generate one chat response
        print(f"Testing chat response for: {args.test_chat}")
        response = handle_chat_query(args.test_chat)
        print("\n===== CHAT RESPONSE =====\n")
        print(response)
    else:
        # Service mode - run continuous loop
        service_loop()