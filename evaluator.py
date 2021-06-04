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


def get_dataset(folder_name):
    training = torch.load(folder_name.joinpath("softmax_training.pt"))
    testing = torch.load(folder_name.joinpath("softmax_validation.pt"))
    batch_size = training.shape[0]
    output_set_training = torch.ones(training.shape[0])
    output_set_testing = torch.zeros(testing.shape[0])

    dataset = torch.cat([training, testing], dim=0)
    labels = torch.cat([output_set_training, output_set_testing], dim=0)

    return dataset, labels


def get_metrics(folder_name):
    dataset, labels = get_dataset(folder_name)
    X_train, X_test, y_train, y_test = sk.train_test_split(dataset.numpy(), labels.numpy(), test_size=0.20,
                                                           random_state=42)
    mlp_clf = MLPClassifier(random_state=1, max_iter=300, batch_size=1024).fit(X_train, y_train)
    mlp_pred = mlp_clf.predict(X_test)


    results = {"Precision": precision_score(mlp_pred, y_test), "Accuracy": accuracy_score(mlp_pred, y_test), "Recall": recall_score(mlp_pred, y_test)}
    return {"metrics": results, "model": mlp_clf}


def get_directory_paths(main_folder):
    p = Path(main_folder)
    folder_list = []

    for sparsity in p.iterdir():
        for sample in sparsity.iterdir():
            folder_list.append(sample)

    return folder_list


if __name__ == '__main__':

    filepath = sys.argv[1]
    aggregated_results = {}

    for folder in tqdm(get_directory_paths(filepath)):
        print('working: ',folder)
        try:
            output = get_metrics(folder)
        except Exception as e:
            print("Failed for {} due to {}".format(folder, e.__str__()))
            continue

        results = folder / "metrics.json"

        results.write_text(dumps(output["metrics"]))
        #dump(output["model"], folder / "model.joblib")

        with open(folder / 'shadow_model.pkl','wb') as fout:
            pickle.dump(output['model'],fout)

        aggregated_results[str(folder)] = output["metrics"]
        print(' Done: ',folder)

    with open("results_" + sys.argv[1] + ".json", "w") as f:
        dump(aggregated_results, f)