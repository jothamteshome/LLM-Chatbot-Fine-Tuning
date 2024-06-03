import torch

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Process the generated responses to remove duplicate sentences and to end on
# a punctuation mark
def process_generated_response(response):
    # List containing possible end punctuation marks
    end_punctuation_marks = [".", "?", "!"]

    # If response ends in punctuation, return the response as is
    if response[-1] in end_punctuation_marks:
        return response
    
    # Find the final location for each punctuation type and take the largest index
    end_loc = max([response.rfind("."), response.rfind("!"), response.rfind("?")])

    # If no end punctuation is found, return the response as is
    if end_loc == -1:
        return response
    
    # Slice response to end on punctuation mark
    response = response[:end_loc+1]


    # Remove duplicate sentences from response
    final_response = []
    existing_sentences = {}

    # Loop through sentences in response
    for sentence in response.split("."):
        # If sentence already exists, continue through loop
        if sentence in existing_sentences or not sentence:
            continue

        # Add sentences to final response
        final_response.append(f"{sentence}.")

    # Return joined final response
    return "".join(final_response)


# Generates a single response using the given chat history
def generate_response(model, tokenizer, chat):
    # Set up the pipeline for a text generation task
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate a response using the chat history
    response = pipe(chat, max_new_tokens=150)

    # Process and replace generated response output
    processed_response_content = process_generated_response(response[0]['generated_text'][-1]['content'])
    response[0]['generated_text'][-1]['content'] = f"{tokenizer.bos_token}{processed_response_content}{tokenizer.eos_token}"

    # Print the most recent response
    print(response[0]['generated_text'][-1]['content'])

    return response[0]['generated_text']


# Function to handle running the inference on given chat messages
def run_inference(args):
    # Load in the fine-tuned model and tokenizer
    try:
        # Try loading ONNX Runtime model first
        model = ORTModelForCausalLM.from_pretrained(args.model_loc, torch_dtype=torch.bfloat16)
    except FileNotFoundError:
        # Fall back on regular AutoModel if ONNX Runtime model not found
        model = AutoModelForCausalLM.from_pretrained(args.model_loc, torch_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_loc)

    # Initialize the chat with a generic system message
    chat = []

    # Loop forever
    while True:
        # Add the current user input message to the chat history
        chat.append({'role': "user", "content": f'{tokenizer.bos_token}{input(">> ")}{tokenizer.eos_token}'})

        # Run inference on the current chat history and update the chat history
        # for future inference
        chat = generate_response(model, tokenizer, chat)