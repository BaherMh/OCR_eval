# OCR Evalution Repo
this repo is to facilitate evaluating OCR moodels, with different models added, benchmarks and evaluation metrics

## Quick Start
to use the model to inference on a apecific dataset you can run
```bash
git clone https://github.com/BaherMh/OCR_Ru.git
pip install requirements.txt
python .\run.py --dataset {dataset_name} --model {model_name}
```

Example
```bash
git clone https://github.com/BaherMh/OCR_Ru.git
pip install requirements.txt
python .\run.py --dataset RusTwit --model ppocr_v5_mobile
```

To Evaluate multiple models on multiple benchmarks with one command
```bash
git clone https://github.com/BaherMh/OCR_Ru.git
pip install requirements.txt
python .\run.py --dataset {dataset_name1} {dataset_name1} --model {model_name1} {model_name2} {model_name3}
```
this will evalaute all the possible combinations and save the results

Alternatively for debugging you can activate the debug model, this will only evaluate the first 5 samples of dataset

Example
```bash
git clone https://github.com/BaherMh/OCR_Ru.git
pip install requirements.txt
python .\run.py --dataset RusTwit --model ppocr_v5_mobile --debug
```

## Add a new model
this repo faciliate the work of adding a new model and evaluating it, you should follow these steps:

1- add a new class that enhirits from BaseOCR in the methods package

2- implement run_method() which takes a path to some image, and outputs its predicted text

3- add you new model to the dictionary of supported models in config.py, with the key being the name you want to refer to the new model during evaluation

Done! now you can easily evaluate new model!


## Add a new dataset

it is also super easy to add your new dataset

1- create .tsv file of your dataset, which should include two columns, 'image' which includes the base64 encoding of the image, and 'answer' that includes the text of the image 

2- add the path to your .tsv dataset to the dataset_paths dictionary in config.py, with the key being the name you want to refer to the new data during evaluation

Done! you can now evalaute anu of the supported models on this new data
