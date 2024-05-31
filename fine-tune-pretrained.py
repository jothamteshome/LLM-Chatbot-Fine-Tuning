import huggingface_hub
import torch

from load_datasets import load_split_dataset
from optimum.onnxruntime import ORTModelForCausalLM
from run_inference import run_inference
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, TrainingArguments   
from trl import SFTTrainer, setup_chat_format

# Set transformers logging verbosity
logging.set_verbosity_error()

# Function to handle the fine-tuning of the model using a given dataset
def run_training(dataset_name, model_name):
    # Load pretrained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add new tokens when running on bitext_customer support dataset
    if dataset_name == "bitext_customer_support":
        # Download entities file from repository
        huggingface_hub.hf_hub_download(repo_id="jothamteshome/customerSupportChatbot",
                                    filename="bitext_customer_support_entities.txt",
                                    local_dir="./")
        
        with open("bitext_customer_support_entities.txt", "r") as f:
            new_tokens = "".join(f.readlines()).strip().split("\n")

        # Set new tokens
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

    # Export model to onnx
    export_model_to_onnx(dataset_name, model_name)


# Export transformers model to ONNX for better cpu inference performance
def export_model_to_onnx(dataset_name, model_name):
    model_name = model_name.split("/")[-1]

    # Load existing fine-tuned model
    ort_model = ORTModelForCausalLM.from_pretrained(f"{dataset_name}-{model_name}-model/tuned_model", export=True)
    tokenizer = AutoTokenizer.from_pretrained(f"{dataset_name}-{model_name}-model/tuned_model")

    # Export model using onnx
    ort_model.save_pretrained(f"{dataset_name}-{model_name}-model/onnx_model")
    tokenizer.save_pretrained(f"{dataset_name}-{model_name}-model/onnx_model")


if __name__ == "__main__":
    dataset_name = "bitext_customer_support"
    model_name = "microsoft/DialoGPT-medium"
    # run_training(dataset_name, model_name)
    run_inference(dataset_name, model_name)