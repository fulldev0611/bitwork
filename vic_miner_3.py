import os
import argparse
from typing import Dict, List
from flask import Flask, request, jsonify
import json
import random
import traceback
import sys
import time
import requests
import re
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run API with OpenAI parameters.")
    parser.add_argument("--auth_token", default="paul123", help="Authentication token")
    parser.add_argument("--model_name", type=str, default= "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ", help="Model name")
    parser.add_argument("--port", default=8091, type=int, help="Port Number")
    return parser.parse_args()


# Define the Flask app
app = Flask(__name__)

@app.route("/", methods=["POST"])
def chat():
    # Check authentication token
    request_data = request.get_json()
    auth_token = request_data.get("verify_token")
    if auth_token != args.auth_token:
        return jsonify({"error": "Invalid authentication token"}), 401

    # Get messages from the request

    history = request_data.get("history", [])
    n = request_data.get('n', 1)
    # Call the forward function and get the response
    try:
        response = miner.forward(history, num_replies = n)
    except:
        traceback.print_exc(file=sys.stderr)
        return "An error occured"
    if len(response) == 1:
        response = response[0]
    # Return the response
    return jsonify({"response": response})


class ModelMiner():

    def __init__( self, model_name, device="cuda", max_length=350, temperature=0.73, do_sample=True ):
        super( ModelMiner, self ).__init__()
        
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True )
        self.model = AutoModelForCausalLM.from_pretrained( model_name, device_map="auto", revision="main", trust_remote_code=True )
        print("model loaded")
                        
        if self.device != "cpu":
            self.model = self.model.to( self.device )

    
    def forward(self, history, num_replies=4):

        # history = self._process_history(messages)

        print("History: " + str(history))

        prompt = history + "ASSISTANT:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        output = self.model.generate(
            inputs=input_ids,
            max_new_tokens=512,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_k=40,
            top_p=0.95,
            max_new_tokens=512
        )
 

        completion = self.tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )
                
        # Logging input and generation if debugging is active
        print("Message: " + str(history),flush=True)
        print("Generation: " + str(completion),flush=True)

        return completion

if __name__ == "__main__":
    args = parse_arguments()
    miner = ModelMiner(args.model_name)
    app.run(host="0.0.0.0", port=args.port, threaded=False)

    # pip install protobuf
    # pip install flask
    # pip install sentencepiece
    # pip install Accelerate
