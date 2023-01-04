# ZAC2022-LightningConformer
Team Lightning's solution for Zalo AI Challenge 2022

# Dataset
Link: 
```
https://institute.vinbigdata.org/events/vinbigdata-chia-se-100-gio-du-lieu-tieng-noi-cho-cong-dong/
```
Note that users should create a metadata  file in `.json` format with `audio_filepath`, `transcript` and `note` metadata fields.

# Prepare environment
```
pip install -r requirements.txt
```

# Train the model
We pre-trained the model with self-supervised approach on Vin BigData ASR
```
python train.py --config-name pretrain.yaml
```
After that, this command is to fine-tune the model with ZAC2022 train dataset (Optional)
```
python train.py --config-name finetune.yaml
```

# Inference
Model Inference can be simply performed with one line
```
python inference.py
```
