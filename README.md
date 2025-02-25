# Balancing the Budget

Code for the paper Balancing the Budget: Understanding Trade-offs Between Supervised and
Preference-Based Finetuning


Link - [https://arxiv.org/pdf/2502.11284](https://arxiv.org/pdf/2502.11284)

## Data

Process all the data 

```bash
bash tuning/data_processing.sh
```

## Run

1. Edit the ```train_sizes``` list in ```tuning/run.sh``` to add different #train examples to train the models.
2. Run ```bash tuning/run.sh``` and select the task, sft-pft ratio and base model. 

