import math
import torch

from trl import SFTTrainer, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments   
from datasets import load_dataset, Dataset


# Load the General-Knowledge dataset in the correct format
def load_general_knowledge_dataset():
    # Load the training split of the General-Knowledge dataset from HuggingFace datasets
    dataset = load_dataset("MuskumPillerum/General-Knowledge", split='train').to_pandas()

    # Drop all rows with None values in dataset
    dataset.dropna(inplace=True)

    # Format data into typical format used by trainer
    data = []

    # Add questions as user messages and answers as assistant messages
    for row in dataset.itertuples():
        row_data = {"messages": [], "prompt_id": str(row[0])}
        row_data["messages"].append({"role": "user", "content": row[2]})
        row_data["messages"].append({"role": "assistant", "content": row[1]})
        data.append(row_data)

    # Convert from dictionary back to Huggingface Dataset
    dataset = Dataset.from_list(data)

    # Split the dataset into a training and validation set
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    return dataset


# Load the Ultrachat 200k dataset in the correct format
def load_ultrachat_dataset():
    # Load the training split of the Ultrachat 200k dataset from HuggingFace datasets
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    # Split the dataset into a training and validation set
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    return dataset


# Returnds the directory and dataset corresponding to the dataset name passed in
def load_split_dataset(dataset_name, model_name):
    model_name = model_name.split("/")[-1]

    datasets = {'general_knowledge': load_general_knowledge_dataset, 'ultrachat_200k': load_ultrachat_dataset}
    
    return f"{dataset_name}-{model_name}-model", datasets[dataset_name]()


# Function to handle the fine-tuning of the DistilGPT2 model using a given dataset
def run_training(dataset_name, model_name):
    # Load pretrained model and tokenizer for DistilGPT2
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up the chat format for the model
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Load in the dataset using a given name
    directory, dataset = load_split_dataset(dataset_name, model_name)

    # Set training arguments for the trainer
    training_args = TrainingArguments(directory,
                                        eval_strategy='epoch',
                                        learning_rate=5e-4,
                                        weight_decay=1e-5,
                                        logging_strategy="no",
                                        save_steps=5000)

    # Set up the supervised fine-tuning trainer from TRL
    trainer = SFTTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        train_dataset=dataset['train'],
                        eval_dataset=dataset['test'],
                        max_seq_length=512)


    # Train the model
    trainer.train()

    # Evaluate performance using perplexity
    evaluation_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(evaluation_results['eval_loss']):.2f}")

    # Save the final fine-tuned model
    trainer.save_model(f"{directory}/tuned_model")


# Generates a single response using the given chat history
def generate_response(model, tokenizer, chat):
    # Set up the chat format for the given model
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Set up the pipeline for a text generation task
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate a response using the chat history
    response = pipe(chat, max_new_tokens=128, truncation=True)

    # Print the most recent response
    print(response[0]['generated_text'][-1]['content'])

    return response


# Function to handle running the inference on given chat messages
def run_inference(dataset_name, model_name):
    # Slice model name to obtain directory names
    model_name = model_name.split("/")[-1]
    
    # Load in the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(f"{dataset_name}-{model_name}-model/tuned_model", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(f"{dataset_name}-{model_name}-model/tuned_model")

    # Initialize the chat with a generic system message
    chat = [{"role":"system", "content": "You are very helpful chatbot who responds one answer at a time."}]

    # Loop forever
    while True:
        # Add the current user input message to the chat history
        chat.append({'role': "user", "content": input(">> ")})

        # Run inference on the current chat history and update the chat history
        # for future inference
        chat = generate_response(model, tokenizer, chat)[0]['generated_text']


if __name__ == "__main__":
    dataset_name = "ultrachat_200k"
    model_name = "distilbert/distilgpt2"
    run_training(dataset_name, model_name)
    # run_inference(dataset_name)