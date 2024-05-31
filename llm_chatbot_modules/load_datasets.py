from convokit import Corpus, download
from datasets import load_dataset, Dataset
from natsort import index_natsorted

# Search through and format single conversation between two characters
def search_conversation(utterances, conv_id):
    # Initialize conversation data object
    conv_data = {"messages": [], "prompt_id": str(conv_id)}

    # Select utterances with the current conversation id
    conversation = utterances[utterances['conversation_id'] == conv_id]

    # Use first speaker as "user" and second speaker as "assistant" 
    user_message = True

    # Loop through utterances and add to conversation data object
    for utterance in conversation.itertuples():
        # Append conversation data
        conv_data['messages'].append({"role": "user" if user_message else "assistant", "content": utterance[2]})

        # Flip user message state
        user_message = not user_message

    return conv_data


# Load the Movie-Dialogs Corpus in the correct format
def load_movie_dialog_dataset():
    # Load corpus using convokit
    corpus = Corpus(filename=download("movie-corpus"))

    # Load utterance data as pandas dataframe
    utterances = corpus.get_utterances_dataframe()

    # Sort utterances by id
    utterances = utterances.iloc[index_natsorted(utterances.index)]

    # Find all unique conversation ids
    conv_ids = utterances.conversation_id.unique()

    # Store conversation data in list
    data = []

    # Loop through conversation ids and build dataset
    for conv_id in conv_ids:
        data.append(search_conversation(utterances, conv_id))

    # Convert from list of dictionaries back to Huggingface Dataset
    dataset = Dataset.from_list(data)

    return dataset


# Load the Bitext Customer Support dataset in the correct format
def load_bitext_customer_support_dataset():
    # Load the training split of the Bitext Customer Support dataset from HuggingFace datasets
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split='train').to_pandas()

    # Format data into typical format used by trainer
    data = []

    # Add questions as user messages and answers as assistant messages
    for row in dataset.itertuples():
        row_data = {"prompt": row[2], "messages": [], "prompt_id": row[0]}
        row_data["messages"].append({"role": "user", "content": row[2]})
        row_data["messages"].append({"role": "assistant", "content": row[5]})
        data.append(row_data)

    # Convert from list of dictionaries back to Huggingface Dataset
    dataset = Dataset.from_list(data)

    return dataset


# Load the Code-Feedback dataset in the correct format
def load_code_feedback_dataset():
    # Load the training split of the Code-Feedback dataset from HuggingFace datasets
    dataset = load_dataset("m-a-p/Code-Feedback", split="train")

    return dataset


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

    # Convert from list of dictionaries back to Huggingface Dataset
    dataset = Dataset.from_list(data)

    return dataset


# Load the Ultrachat 200k dataset in the correct format
def load_ultrachat_dataset():
    # Load the training split of the Ultrachat 200k dataset from HuggingFace datasets
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    return dataset


# Returns the directory and dataset corresponding to the dataset name passed in
def load_split_dataset(dataset_name, model_name):
    model_name = model_name.split("/")[-1]

    datasets = {'general_knowledge': load_general_knowledge_dataset, 
                'ultrachat_200k': load_ultrachat_dataset,
                'code_feedback': load_code_feedback_dataset,
                'movie_dialog_corpus': load_movie_dialog_dataset,
                'bitext_customer_support': load_bitext_customer_support_dataset}
    
    if datasets.get(dataset_name, False):
        return f"fine_tuned_models/{dataset_name}-{model_name}-model", datasets[dataset_name]()
    else:
        raise KeyError("Dataset does not exist. Please add function to format dataset in `load_datasets.py`.")