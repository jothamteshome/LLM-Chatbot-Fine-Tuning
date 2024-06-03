## Getting Started

### Setup

First, make sure you have `Python>=3.8` installed on your system.

Then, run the following to install the required packages
```
pip install -r requirements.txt
```


### Arguments

To get started, here are the arguments to begin running the scripts from the command line

#### Run Script Args
```
usage: main.py [-h] {tune,infer} ...

Fine-tune a pretrained model or generate text with a pretrained/existing model    

positional arguments:
  {tune,infer}  Options to run fine-tuning or inference on model
    tune        Fine-tune a Hugging Face pretrained model
    infer       Run inference on a Hugging Face pretrained model or existing model

options:
  -h, --help    show this help message and exit
```

#### Fine-Tuning Arguments
```
usage: main.py tune [-h] [-m MODEL_NAME] [-d {bitext_customer_support,code_feedback,general_knowledge,movie_dialog_corpus,ultrachat_200k}] [-e EPOCHS] [-wd WEIGHT_DECAY] [-lr LEARNING_RATE]

options:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model MODEL_NAME
                        Local directory or Hugging Face Repo containing model (default: microsoft/DialoGPT-medium)
  -d {bitext_customer_support,code_feedback,general_knowledge,movie_dialog_corpus,ultrachat_200k}, --dataset {bitext_customer_support,code_feedback,general_knowledge,movie_dialog_corpus,ultrachat_200k}
                        Name of dataset to load from `load_datasets.py` (default: general_knowledge)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to tune the model for (default: 3)
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        Value of the weight decay to apply to layers in optimizer (default: 0.01)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Initial learning rate to use in optimizer (default: 0.0005)
```

#### Inference Arguments
```
usage: main.py infer [-h] [-m MODEL]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Local directory or Hugging Face Repo containing model (default: jothamteshome/customerSupportChatbot)
```