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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import bittensor


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run API with OpenAI parameters.")
    parser.add_argument("--auth_token", default="paul123", help="Authentication token")
    parser.add_argument("--model_name", type=str, default= "cerebras/btlm-3b-8k-base", help="Model name")
    parser.add_argument("--port", default=8093, type=int, help="Port")

    parser.add_argument(
            "--cerebras.device", type=str, help="Device to load model", default="cuda"
    )
    parser.add_argument(
            "--cerebras.max_length",
            type=int,
            default=50,
            help="The maximum length (in tokens) of the generated text.",
    )
    parser.add_argument(
            "--cerebras.do_sample",
            action="store_true",
            default=False,
            help="Whether to use sampling or not (if not, uses greedy decoding).",
    )
    parser.add_argument(
            "--cerebras.no_repeat_ngram_size",
            type=int,
            default=2,
            help="The size of the n-grams to avoid repeating in the generated text.",
    )

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

    messages = request_data.get("messages", [])
    n = request_data.get('n', 1)

    print("request_message" + str(messages))
    print("request_n" + str(n))

    # Call the forward function and get the response
    try:
        response = miner.forward(messages)
    except:
        traceback.print_exc(file=sys.stderr)
        return "An error occured"
    if len(response) == 1:
        response = response[0]
    # Return the response
    return jsonify({"response": response})


class ModelMiner():

    def __init__( self, max_length=50, no_repeat_ngram_size=2 ):
        super( ModelMiner, self ).__init__()
      
    
        model = AutoModelForCausalLM.from_pretrained(
            "cerebras/btlm-3b-8k-base", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "cerebras/btlm-3b-8k-base",
            trust_remote_code=True,
        )

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0,
            do_sample=False,
            max_new_tokens=250,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    @staticmethod
    def _process_history(history: List[Dict[str, str]]) -> str:
        processed_history = ""
        for message in history:
            if message["role"] == "system":
                processed_history += "system: " + message["content"] + "\n"
            if message["role"] == "assistant":
                processed_history += "assistant: " + message["content"] + "\n"
            if message["role"] == "user":
                processed_history += "user: " + message["content"] + "\n"
        return processed_history
    
    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self._process_history(messages)
        generation = self.pipe(history)[0]["generated_text"].split(":")[-1].replace(str(history), "")

        print("History:  " + str(history))        
        print("Message: " + str(messages),flush=True)
        print("Generation: " + str(generation),flush=True)

        return (
            self.pipe(history)[0]["generated_text"]
            .split(":")[-1]
            .replace(str(history), "")
        )

    

if __name__ == "__main__":
    args = parse_arguments()
    miner = ModelMiner(args.model_name)
    app.run(host="0.0.0.0", port=args.port, threaded=False)
