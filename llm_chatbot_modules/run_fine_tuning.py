import torch

from huggingface_hub import hf_hub_download
from llm_chatbot_modules.load_datasets import load_split_dataset
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, TrainingArguments   
from trl import SFTTrainer, setup_chat_format

# Set transformers logging verbosity
logging.set_verbosity_error()

# Function to handle the fine-tuning of the model using a given dataset
def run_fine_tuning(args):
    # Load pretrained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'bos_token': "<|im_start|>", "eos_token": "<|im_end|>"})

    # Add new tokens when running on bitext_customer support dataset
    if args.dataset == "bitext_customer_support":
        # Download entities file from repository
        hf_hub_download(repo_id="jothamteshome/customerSupportChatbot",
                            filename="bitext_customer_support_entities.txt",
                            local_dir="huggingface_files")
        
        with open("huggingface_files/bitext_customer_support_entities.txt", "r") as f:
            new_tokens = "".join(f.readlines()).strip().split("\n")

        # Set new tokens
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(tokenizer))


    # Set up the chat format for the model
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Load in the dataset using a given name
    directory, dataset = load_split_dataset(args.dataset, args.model_name, tokenizer)

    # Set training arguments for the trainer
    training_args = TrainingArguments(directory,
                                        num_train_epochs=args.epochs,
                                        learning_rate=args.learning_rate,
                                        weight_decay=args.weight_decay,
                                        per_device_train_batch_size=1,
                                        gradient_accumulation_steps=4,
                                        bf16=True,
                                        disable_tqdm=False,
                                        eval_strategy="steps",
                                        save_strategy="steps",
                                        logging_strategy="epoch",
                                        eval_steps=1000,
                                        save_steps=1000,
                                        load_best_model_at_end=True,
                                        save_total_limit=1
                                        )

    # Set up the supervised fine-tuning trainer from TRL
    trainer = SFTTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        train_dataset=dataset['train'],
                        eval_dataset=dataset['test'],
                        max_seq_length=512)


    # Train the model
    trainer.train()

    # Save the final fine-tuned model
    trainer.save_model(f"{directory}/tuned_model")

    # Export model to onnx
    export_model_to_onnx(directory)


# Export transformers model to ONNX for better cpu inference performance
def export_model_to_onnx(directory):
    # Load existing fine-tuned model
    ort_model = ORTModelForCausalLM.from_pretrained(f"{directory}/tuned_model", export=True)
    tokenizer = AutoTokenizer.from_pretrained(f"{directory}/tuned_model")

    # Export model using onnx
    ort_model.save_pretrained(f"{directory}/onnx_model")
    tokenizer.save_pretrained(f"{directory}/onnx_model")