import os
import json
import time
import random
import logging
import requests
from openai import OpenAI

def setup_caller_logger():
    """Set up logger for model caller"""
    logger = logging.getLogger('model_caller')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_caller_logger()

def call_api_with_retry(client_info, messages, max_tokens=2048, temperature=0.2, thinking=None, top_p=1.0, max_retries=10):
    """Unified API calling function, supports two main model types"""
    model_type = client_info.get("model_type", "openai")
    model_path = client_info.get("model_path", "")
    
    attempt = 0
    last_exception = None
    
    while attempt < max_retries:
        try:
            attempt += 1
            if attempt > 1:
                logger.info(f"Attempting API call {attempt}/{max_retries}...")
            
            # Call different APIs based on model type
            if model_type == "openai":
                client = client_info.get("client")
                response = client.chat.completions.create(
                    messages=messages,
                    model=model_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                return True, response
                
            elif model_type in ["gemini", "claude", "claude-3-7-sonnet-20250219", "deepseek-r1", "deepseek-r1-inner", "gpt-4o"]:
                gemini_messages = []
                for message in messages:
                    if isinstance(message["content"], list):
                        # Handle OpenAI format messages
                        content = ""
                        for part in message["content"]:
                            if part['type'] == "text":
                                content = part['text']
                        gemini_message = {"role": message["role"], "content": content}
                    else:
                        # Handle messages with simple string content
                        gemini_message = {"role": message["role"], "content": message["content"]}
                    gemini_messages.append(gemini_message)

                headers = {
                    'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}',
                    'Content-Type': 'application/json'
                }
                
                # Use model_type as actual model name to pass to API
                actual_model = model_path if model_path != "deepseek-r1-inner" else "deepseek-r1"
                
                payload = json.dumps({
                    "model": model_type,
                    "messages": gemini_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                })

                response = requests.request(
                    "POST", 
                    "url",  # Replace with actual API URL
                    headers=headers, 
                    data=payload
                )


                if response.status_code == 200:
                    if model_type == "deepseek-r1" or model_type == "deepseek-r1-inner" :
                        return True, response.json()['choices'][0]['message']['content'], response.json()['choices'][0]['message']['reasoning_content']
                    elif model_type == "claude-3-7-sonnet-20250219" or model_type == "claude":
                        return True, response.json()['choices'][0]['message']['content']
                    else:
                        return True, response.json()['choices'][0]['message']['content']
                else:
                    error_info = response.json()
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        # Add disclaimer
                        last_msg = messages[-1]
                        if isinstance(last_msg["content"], list):
                            if not last_msg["content"][0]["text"].endswith("They do not represent any real events or entities. ]"):
                                last_msg["content"][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                        else:
                            if not last_msg["content"].endswith("They do not represent any real events or entities. ]"):
                                last_msg["content"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value
                    raise Exception(f"API call failed: {code_value}")
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            last_exception = e
            logger.warning(f"API call failed (attempt {attempt}/{max_retries}): {e}")
            
            # Exponential backoff strategy
            wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
            logger.info(f"Waiting {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)
    
    logger.error(f"Reached the maximum number of retries ({max_retries}), giving up.")
    return False, str(last_exception)

def initialize_client(model_info):
    """Initialize and return client information"""
    model_path = model_info.get("model_path", "")
    
    # Support both deepseek-r1 and deepseek-r1-inner
    if (model_path.startswith("gemini") or 
          model_path.startswith("claude") or 
           model_path.startswith("gpt-4o") or
          model_path.startswith("claude-3-7-sonnet-20250219") or
          model_path.startswith("deepseek-r1") or 
          model_path.startswith("deepseek-r1-inner")):
        
        # Use model name directly as type
        if model_path.startswith("gemini"):
            model_type = "gemini"
        elif model_path.startswith("claude") or model_path.startswith("claude-3-7-sonnet-20250219"):
            model_type = "claude-3-7-sonnet-20250219"
        elif model_path.startswith("deepseek-r1") or model_path.startswith("deepseek-r1-inner"):
            model_type = "deepseek-r1-inner"
        elif model_path.startswith("gpt-4o"):
            model_type = "gpt-4o"
        else:
            model_type = model_path  # deepseek-r1 or deepseek-r1-inner
        
        if model_path == 'gemini-2.0-flash-thinking-exp':
            model_path = 'gemini-2.0-flash-thinking-exp-1219'
            
        return {
            "model_type": model_type,
            "model_path": model_path
        }
    
    else:
        # Default to local OpenAI interface, initialize client
        try:
            client = OpenAI(api_key="0", base_url="url")  # Modify for different tasks
            return {
                "model_type": "openai",
                "model_path": model_path,
                "client": client
            }
        except Exception as e:
            logger.error(f"OpenAI client initialization failed: {e}")
            raise