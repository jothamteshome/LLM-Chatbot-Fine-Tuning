import argparse

from llm_chatbot_modules.run_fine_tuning import run_fine_tuning
from llm_chatbot_modules.run_inference import run_inference


# Create a subparser for running fine tuning on pretrained model
def create_fine_tuning_parser(subparsers):
    # Subparser to handle arguments for model fine-tuning tasks
    fine_tuning_parser = subparsers.add_parser('fine-tune', help="Fine-tune a Hugging Face pretrained model")

    # Add model name argument to fine-tuning parser
    fine_tuning_parser.add_argument("--model_name",
                                    help="Local directory or Hugging Face Repo containing model (default: %(default)s)",
                                    default="microsoft/DialoGPT-medium",
                                    required=False)
    
    # Add dataset argument to fine-tuning parser
    fine_tuning_parser.add_argument("--dataset",
                                    choices=["bitext_customer_support", "code_feedback", "general_knowledge", "movie_dialog_corpus", "ultrachat_200k"],
                                    help="Name of dataset to load from `load_datasets.py` (default: %(default)s)",
                                    default="general_knowledge",
                                    required=False)
    
    # Add epochs argument to fine-tuning parser
    fine_tuning_parser.add_argument("--epochs",
                                    help="Number of epochs to tune the model for (default: %(default)s)",
                                    default=4,
                                    type=int,
                                    required=False)
    
    # Add weight decay value argument to fine-tuning parser
    fine_tuning_parser.add_argument("--weight_decay",
                                    help="Value of the weight decay to apply to layers in optimizer (default: %(default)s)",
                                    default=1e-2,
                                    type=float,
                                    required=False)
    
    # Add learning rate value argument to fine-tuning parser
    fine_tuning_parser.add_argument("--learning_rate",
                                    help="Initial learning rate to use in optimizer (default: %(default)s)",
                                    default=5e-5,
                                    type=float,
                                    required=False)
    
    
    # Add function default for fine-tuning parser
    fine_tuning_parser.set_defaults(func=run_fine_tuning)


# Create a subparser for running inference on pretrained or fine-tuned model
def create_inference_parser(subparsers):
    # Subparser to handle arguments for model inference tasks
    inference_parser = subparsers.add_parser("infer", help="Run inference on a Hugging Face pretrained model or existing model")
    inference_parser.add_argument("--model_loc",
                                  help="Local directory or Hugging Face Repo containing model (default: %(default)s)",
                                  default="jothamteshome/customerSupportChatbot",
                                  required=False)
    
    # Add function default for inference parser
    inference_parser.set_defaults(func=run_inference)


def main():
    # Create argument parser to handle running of appropriate files
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained model or generate text with a pretrained/existing model")    

    # Add subparsers to initial parser
    subparsers = parser.add_subparsers(help="sub-command-help", required=True)
    create_fine_tuning_parser(subparsers)
    create_inference_parser(subparsers)

    # Parse arguments and run correct file
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()