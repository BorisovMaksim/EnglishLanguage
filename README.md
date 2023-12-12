# This is a solution to LightAutoML course practice 

I compared lightautoml solution with fairseqsolution  in text classification task. Data is taken from kaggle competition:
https://www.kaggle.com/competitions/feedback-prize-effectiveness

# LightAutoML

- See notebook AutoML.ipynb

# Fairseq 
I fine-tuned roberta-base model on training data. To reproduce:
- install fairseq 
- ``pip install requirements.txt``
- ``python preprocess_data.py``
- ``bash process.sh``
- ``bash train.sh``
- ``python inference_roberta.py``

# Results

| Model | framework |f-score for 'Adequate'  | f-score for 'Effective'    | f-score for 'Ineffective'    | Mean f-score |
| :---:   | :---:  | :---: | :---: |  :---: |  :---: | 
| bert-tiny | lightautoml| 0.7360   | 0.6379   |    0.0000     | 0.4579 |
| bert-base-uncased | lightautoml |0.7472   |  0.6792   |   0.0000   | 0.4755|
| roberta-base | fairseq | 0.7598  |  0.6238  |   0.3012     | 0.5616 | 

