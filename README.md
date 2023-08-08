# Dermatology-image-classification
A novel real-time DNN framework for dermatology image classification task on low-power embedded devices.

## Prerequiste
Run the requirements.txt to install required packages
```
pip install requirements.txt
```

## Commands
To train and test the DNN, ensure that the data is available with the permission from ESFair 2023 (https://esfair2023.github.io/ESFair/)

```
python .\main.py --type all
```
Additional settings include:
* Version (1-5): indicates the way of partitioning the training and testing sets with 5 cross-validation techniques.
* batch_train & batch_test: cutomize the batch size for training and testing respectively.
* Seed: controls the random values for reproducing experiments
* sample_size(0-5): the targeted data sizes for each class (0-5). Default is 0.
* std: Standard deviation for Gaussian white noise.
* epoch: Training iterations
* new_data: indicator for generating new training and testing data or using the existing sets. The data filenames for each set store in log/(test/train)_samples_(subgroup).txt

An example with all the parameters shows below:
```
python .\main.py --type test --version 1 --batch_train 16 --batch_test 16 --seed 123 --sample_size0 4000 --sample_size1 4000 --sample_size2 4000 --sample_size3 4000 --sample_size4 2500 --sample_size5 2500 --std 0.03 --epochs 400 --new_data false
```

If you want to test the trained model with a testing set,
```
python main.py --final true --final_model <path-to-model> --testing_data_path <testing-path> --batch 32
```
