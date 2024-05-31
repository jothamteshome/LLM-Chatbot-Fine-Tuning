from llm_chatbot_modules.run_fine_tuning import run_fine_tuning
from llm_chatbot_modules.run_inference import run_inference


def main():
    # Set dataset name and model name to use
    dataset_name = "bitext_customer_support"
    model_name = "microsoft/DialoGPT-medium"

    # Fine tune selected model using selected dataset
    # Only datasets found in `load_datasets.py` can be used at the moment
    # run_fine_tuning(dataset_name, model_name)

    # Pass in local directory or huggingface repo location of model to run inference
    # Defaults to `jothamteshome/customerSupportChatbot`
    run_inference()


if __name__ == "__main__":
    main()