from load_datasets import load_split_dataset
from torch import bfloat16
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments   
from trl import SFTTrainer, setup_chat_format


# Function to handle the fine-tuning of the model using a given dataset
def run_training(dataset_name, model_name):
    # Load pretrained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add new tokens when running on bitext_customer support dataset
    if dataset_name == "bitext_customer_support":
        with open("bitext_customer_support_tokens.txt", "r") as f:
            new_tokens = "".join(f.readlines()).strip().split("\n")

        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(tokenizer))


    # Set up the chat format for the model
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Load in the dataset using a given name
    directory, dataset = load_split_dataset(dataset_name, model_name)

    # Set training arguments for the trainer
    training_args = TrainingArguments(directory,
                                        eval_strategy='no',
                                        num_train_epochs=7,
                                        learning_rate=5e-5,
                                        weight_decay=1e-5,
                                        per_device_train_batch_size=1,
                                        gradient_accumulation_steps=4,
                                        bf16=True,
                                        logging_strategy="epoch",
                                        save_total_limit=1,
                                        )

    # Set up the supervised fine-tuning trainer from TRL
    trainer = SFTTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        train_dataset=dataset,
                        max_seq_length=1024)


    # Train the model
    trainer.train()

    # Save the final fine-tuned model
    trainer.save_model(f"{directory}/tuned_model")


# Generates a single response using the given chat history
def generate_response(model, tokenizer, chat):
    # Set up the chat format for the given model
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Set up the pipeline for a text generation task
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate a response using the chat history
    response = pipe(chat, max_new_tokens=32, truncation=True)

    # Print the most recent response
    print(response[0]['generated_text'][-1]['content'])

    return response


# Function to handle running the inference on given chat messages
def run_inference(dataset_name, model_name):
    # Slice model name to obtain directory names
    model_name = model_name.split("/")[-1]
    
    # Load in the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(f"{dataset_name}-{model_name}-model/tuned_model", torch_dtype=bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(f"{dataset_name}-{model_name}-model/tuned_model")

    # Make sure that model token embeddings contain any new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Initialize the chat with a generic system message
    chat = []

    # Loop forever
    while True:
        # Add the current user input message to the chat history
        chat.append({'role': "user", "content": input(">> ") + tokenizer.eos_token})

        # Run inference on the current chat history and update the chat history
        # for future inference
        chat = generate_response(model, tokenizer, chat)[0]['generated_text']


if __name__ == "__main__":
    dataset_name = "bitext_customer_support"
    model_name = "microsoft/DialoGPT-medium"
    run_training(dataset_name, model_name)
    # run_inference(dataset_name, model_name)