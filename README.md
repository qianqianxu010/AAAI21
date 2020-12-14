# AAAI21
Pytorch implementation for AAAI21 paper: Deep Partial Rank Aggregation for Personalized Attributes


## Run
Command:
```python mle_train.py [train_file] [test_file] [save_path] [log_file] [gpu_id] [backbone]```

Parameters:
- `train_file`: training annotation file, e.g., 'lfw10/lfw10_train.txt'
- `test_file`: test annotation file, e.g., 'lfw10/lfw10_test.txt'
- `save_path`: model save path
- `log_file`: output log file path
- `gpu_id`: single id supported
- `backbone`: 'alexnet', 'vgg', 'resnet'

Competitor: `naive_train.py`
