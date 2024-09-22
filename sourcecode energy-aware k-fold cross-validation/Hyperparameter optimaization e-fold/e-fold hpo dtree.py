import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

k = 10

criterion_list = ['gini', 'entropy']
max_depth_factor_list = [1, 10, 20, 30]
min_samples_split_list = [2, 6, 10, 14, 18, 22, 26, 30]

overall_all_scores_efold = {}
overall_all_scores_kfold = {}
overall_folds_needed = {}

overall_best_params_efold = {}
overall_best_params_kfold = {}

best_test_scores_efold = {}
best_test_scores_kfold = {}
best_test_scores_diff = {}

for iteration in range(1, 26):
    random_state_value = iteration

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_value)

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state_value)

    run = 0

    best_score_efold = 0
    best_score_kfold = 0
    best_params_efold = {}
    best_params_kfold = {}
    all_scores_efold = {}
    all_scores_kfold = {}
    folds_needed = {}

    # Manuelle Hyperparameter-Optimierung
    for criterion in criterion_list:
        for max_depth in max_depth_factor_list:
            for min_samples_split in min_samples_split_list:

                all_fold_scores = {}  # Die Scores der jeweiligen CV pro Fold
                all_folds_meanscore = {}  # Die Mean Scores der CV zum jeweiligen Fold
                std_after_each_fold = {}

                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                               min_samples_split=min_samples_split, random_state=random_state_value)
                check = True
                std_count = 0

                run = run + 1
                print(f"Run: {run}")
                print(
                    f"Hyperparameter: criterion={criterion}, max_depth={max_depth}, min_samples_split={min_samples_split}")

                for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train), start=1):
                    X_CV_train, X_CV_test = X_train[train_idx], X_train[test_idx]
                    y_CV_train, y_CV_test = y_train[train_idx], y_train[test_idx]

                    model.fit(X_CV_train, y_CV_train)
                    model_score = f1_score(y_CV_test, model.predict(X_CV_test))

                    all_fold_scores[fold_idx] = model_score

                    mean_scores = np.mean(list(all_fold_scores.values()))
                    all_folds_meanscore[fold_idx] = mean_scores

                    if fold_idx > 1:
                        # Berechnung der Standardabweichung Scores nach jedem Fold
                        std = np.std(list(all_fold_scores.values()),
                                     ddof=1)
                        std_after_each_fold[fold_idx] = std

                    if fold_idx > 2:
                        actual_std = std  # aktuelle std
                        last_std = std_after_each_fold[fold_idx - 1]  # vorherige std
                        # last_last_std = std_after_each_fold[fold_idx-2]

                        if actual_std < last_std:  # wenn die aktuelle kleiner ist als die vorherige erhöhen wir einen counter
                            std_count += 1
                        else:  # wenn nicht prüfen wir wie gravierend der Unterschied ist
                            # Abweichung zum vorherigen gravierend ?
                            std_change = abs(actual_std - last_std)
                            std_change_percent = (std_change / actual_std) * 100
                            if std_change_percent > 5:  # größer als 5% dann count zurücksetzen , kleiner dann erhöhen
                                std_count = 0
                            else:
                                std_count += 1

                        if std_count == 2:  # Wenn 2 mal hintereinander geringere std und die nachfolgende ebenfalls dann aufhören
                            if check == True:
                                print(f"abbruch nach k = {fold_idx}")
                                print(f"Model Performance liegt bei {mean_scores}")
                                if mean_scores > best_score_efold:
                                    best_score_efold = mean_scores
                                    best_params_efold = {'criterion': criterion, 'max_depth': max_depth,
                                                         'min_samples_split': min_samples_split, 'run': run}

                                cancelation_fold = fold_idx
                                folds_needed[run] = cancelation_fold
                                all_scores_efold[run] = {'criterion': criterion, 'max_depth': max_depth,
                                                         'min_samples_split': min_samples_split,
                                                         'mean_score': mean_scores}
                                check = False

                if check == True:
                    cancelation_fold = fold_idx
                    folds_needed[run] = cancelation_fold
                    print(f"Kein vorzeitiger Abbruch")

                print(f"Tatsächlicher meanscroe: {mean_scores}")

                if mean_scores > best_score_kfold:
                    best_score_kfold = mean_scores
                    best_params_kfold = {'criterion': criterion, 'max_depth': max_depth,
                                         'min_samples_split': min_samples_split, 'run': run}
                all_scores_kfold[run] = {'criterion': criterion, 'max_depth': max_depth,
                                         'min_samples_split': min_samples_split, 'mean_score': mean_scores}

    overall_all_scores_efold[iteration] = all_scores_efold
    overall_all_scores_kfold[iteration] = all_scores_kfold
    overall_folds_needed[iteration] = folds_needed

    print(f"\nBeste Hyperparameter e-fold cv: {best_params_efold}")
    print(f"Beste Genauigkeit (Trainingsdaten, Kreuzvalidierung): {best_score_efold:.4f}")
    overall_best_params_efold[iteration] = best_params_efold
    print("_---------------------------------------------------------------")
    print(f"\nBeste Hyperparameter k-fold cv: {best_params_kfold}")
    print(f"Beste Genauigkeit (Trainingsdaten, Kreuzvalidierung): {best_score_kfold:.4f}")
    overall_best_params_kfold[iteration] = best_params_kfold
    print()
    # Checken wie gut die HPOs von e-fold auf ungesehnen Daten klappt
    best_model_efold = DecisionTreeClassifier(criterion=best_params_efold['criterion'],
                                              max_depth=best_params_efold['max_depth'],
                                              min_samples_split=best_params_efold['min_samples_split'],
                                              random_state=random_state_value)
    best_model_efold.fit(X_train, y_train)
    test_score_efold = f1_score(y_test, best_model_efold.predict(X_test))
    best_test_scores_efold[iteration] = test_score_efold
    print(f"Testdaten-Genauigkeit e-fold: {test_score_efold:.4f}")

    # Checken wie gut die HPOs von k-fold auf ungesehnen Daten klappt

    best_model_kfold = DecisionTreeClassifier(criterion=best_params_kfold['criterion'],
                                              max_depth=best_params_kfold['max_depth'],
                                              min_samples_split=best_params_kfold['min_samples_split'],
                                              random_state=random_state_value)
    best_model_kfold.fit(X_train, y_train)
    test_score_kfold = f1_score(y_test, best_model_kfold.predict(X_test))
    best_test_scores_kfold[iteration] = test_score_kfold
    print(f"Testdaten-Genauigkeit k-fold: {test_score_kfold:.4f}")

    percentual_difference = abs(test_score_kfold - test_score_efold)
    percentual_difference_percent = (percentual_difference / test_score_kfold) * 100
    print(f"Prozentuale Differenz zwischen k-fold und e-fold: {percentual_difference_percent:.2f}%")
    best_test_scores_diff[iteration] = percentual_difference_percent

print("Start DB saving")

import json
import pymssql
import pypyodbc as odbc

server = 'LAPTOP-6E2N1SIK\\SQLEXPRESS'  # Korrigierter Servername
database = 'BADB'
driver = 'ODBC Driver 18 for SQL Server'

conn = None
cursor = None

conn = odbc.connect(
    f'DRIVER={{{driver}}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'Trusted_Connection=yes;'
    f'TrustServerCertificate=yes;'
)
cursor = conn.cursor()

method = 'HPO'
algo = 'Decision Tree'
dataset = 'brestcancer'

json_data_overall_folds = json.dumps(overall_folds_needed)
json_data_best_efold = json.dumps(overall_best_params_efold)
json_data_best_kfold = json.dumps(overall_best_params_kfold)
json_data_test_score_efold = json.dumps(best_test_scores_efold)
json_data_test_score_kfold = json.dumps(best_test_scores_kfold)
json_data_test_score_diff = json.dumps(best_test_scores_diff)

cursor.execute('''
    INSERT INTO [dbo].[Hpo] (Methode, Algorithm, Dataset, overall_folds,best_efold,best_kfold,test_score_efold,test_score_kfold,test_score_diff)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (
method, algo, dataset, json_data_overall_folds, json_data_best_efold, json_data_best_kfold, json_data_test_score_efold,
json_data_test_score_kfold, json_data_test_score_diff))

conn.commit()
cursor.close()
conn.close()
print("DB saving complete")