# This is a solution to LightAutoML course practice 

I compared lightautoml solution with fairseq solution and ray tune solution in text classification task. Data is taken from kaggle competition:
https://www.kaggle.com/competitions/feedback-prize-effectiveness

# LightAutoML

- See notebook AutoML.ipynb

# Ray tune + Hugging face

- See notebook RayTune.ipynb

# Fairseq 
I fine-tuned roberta-base model on training data. To reproduce:
- install fairseq 
- ``pip install requirements.txt``
- ``python preprocess_data.py``
- ``bash process.sh``
- ``bash train.sh``
- ``python inference_roberta.py``

# Results

Main metric is log loss


| Model | framework | Log loss|
| :---:   | :---:  | :---: | 
| bert-tiny | lightautoml| 0.8050   | 
| bert-base-uncased | lightautoml |  0.7721  |
| roberta-base | lightautoml | 0.7312  |
| roberta-base | fairseq | 0.7262  |
| roberta-base | ray tune + hugging face | 0.8649  |

