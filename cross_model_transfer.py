from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score
from pathlib import Path
from joblib import dump
from json import dump, dumps
from tqdm import tqdm
import pickle

import sklearn.model_selection as sk
import torch
import sys
import pandas as pd


"""
Cross model attacks.
Take shadow model from 50k and attack everything else.

"""


def get_dataset(folder_name):
    training = torch.load(folder_name.joinpath("softmax_training.pt"))
    testing = torch.load(folder_name.joinpath("softmax_validation.pt"))
    batch_size = training.shape[0]
    output_set_training = torch.ones(training.shape[0])
    output_set_testing = torch.zeros(testing.shape[0])

    dataset = torch.cat([training, testing], dim=0)
    labels = torch.cat([output_set_training, output_set_testing], dim=0)

    return dataset, labels

def get_directory_paths(main_folder):
    p = Path(main_folder)
    folder_list = []

    for sparsity in p.iterdir():
        for sample in sparsity.iterdir():
            folder_list.append(sample)

    return folder_list


def attack(folder_name,model):
    dataset, labels = get_dataset(folder_name)
    X_train, X_test, y_train, y_test = sk.train_test_split(dataset.numpy(), labels.numpy(), test_size=0.20,random_state=42)
    #mlp_clf = pickle.load(open(folder_name / 'shadow_model.pkl','rb'))
    mlp_pred = model.predict(X_test)

    results = {"Precision": precision_score(mlp_pred, y_test), "Accuracy": accuracy_score(mlp_pred, y_test), "Recall": recall_score(mlp_pred, y_test)}
    return results #{"metrics": results}


if __name__ == '__main__':
    
    res18_folders = get_directory_paths('res18_vectors/')
    res50_folders = get_directory_paths('res50_vectors/')

    all_folders = res18_folders + res50_folders

    #Cross transfer attacks only with 50k samples.
    all_folders = [x for x in all_folders if '50000' in x.name]

    all_results = []

    for fol in all_folders:
        source_folder = fol
        source_mlp_clf = pickle.load(open(source_folder / 'shadow_model.pkl','rb'))
        source_results = []

        for tar in all_folders:
            #if tar == fol:
            #    continue
            metrics = attack(tar,source_mlp_clf)
            metrics['source'] = str(source_folder)
            metrics['target'] = str(tar)

            #source_results[str(tar)] = metrics
            source_results.append(metrics)
            all_results.append(metrics)

        with open(source_folder / 'transfer_attack.json','w') as fout:
            dump(source_results,fout)

        print(fol,' done ')

    df = pd.DataFrame(all_results)
    df.to_csv('transfer_results.csv')