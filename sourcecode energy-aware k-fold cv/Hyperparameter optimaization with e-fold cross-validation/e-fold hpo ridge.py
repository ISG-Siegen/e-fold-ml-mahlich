import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

from sklearn.datasets import fetch_california_housing  # (20000 Datensätze)

california = fetch_california_housing()
X = california.data
y = california.target

k = 10

alpha_list = np.logspace(-8, 8, 100)

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

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state_value)

    run = 0

    best_score_efold = np.inf
    best_score_kfold = np.inf
    best_params_efold = {}
    best_params_kfold = {}
    all_scores_efold = {}
    all_scores_kfold = {}
    folds_needed = {}

    for alpha in alpha_list:

        all_fold_scores = {}
        all_folds_meanscore = {}
        std_after_each_fold = {}

        model = Ridge(alpha=alpha)

        check = True
        std_count = 0

        run = run + 1
        print(f"Run: {run}")
        print(f"Hyperparameter: alpha={alpha}")

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_train), start=1):
            X_CV_train, X_CV_test = X_train[train_idx], X_train[test_idx]
            y_CV_train, y_CV_test = y_train[train_idx], y_train[test_idx]

            model.fit(X_CV_train, y_CV_train)
            model_score = mean_absolute_error(y_CV_test, model.predict(X_CV_test))

            all_fold_scores[fold_idx] = model_score

            mean_scores = np.mean(list(all_fold_scores.values()))
            all_folds_meanscore[fold_idx] = mean_scores

            if fold_idx > 1:
                # Berechnung der Standardabweichung Scores nach jedem Fold
                std = np.std(list(all_fold_scores.values()),
                             ddof=1)  # Hier std für Stichproben und nicht der Gesamtmenge
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
                        if mean_scores < best_score_efold:
                            best_score_efold = mean_scores
                            best_params_efold = {'alpha': alpha, 'run': run}

                        cancelation_fold = fold_idx
                        folds_needed[run] = cancelation_fold
                        all_scores_efold[run] = {'alpha': alpha, 'mean_score': mean_scores}

                        check = False

        if check == True:
            cancelation_fold = fold_idx
            folds_needed[run] = cancelation_fold
            print(f"Kein vorzeitiger Abbruch")

        print(f"Tatsächlicher meanscroe: {mean_scores}")

        if mean_scores < best_score_kfold:
            best_score_kfold = mean_scores
            best_params_kfold = {'alpha': alpha, 'run': run}

        all_scores_kfold[run] = {'alpha': alpha, 'mean_score': mean_scores}

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
    best_model_efold = Ridge(alpha=best_params_efold['alpha'])

    best_model_efold.fit(X_train, y_train)
    test_score_efold = mean_absolute_error(y_test, best_model_efold.predict(X_test))
    best_test_scores_efold[iteration] = test_score_efold
    print(f"Testdaten-Genauigkeit e-fold: {test_score_efold:.4f}")

    # Checken wie gut die HPOs von k-fold auf ungesehnen Daten klappt
    best_model_kfold = Ridge(alpha=best_params_kfold['alpha'])

    best_model_kfold.fit(X_train, y_train)
    test_score_kfold = mean_absolute_error(y_test, best_model_kfold.predict(X_test))
    best_test_scores_kfold[iteration] = test_score_kfold
    print(f"Testdaten-Genauigkeit k-fold: {test_score_kfold:.4f}")

    percentual_difference = abs(test_score_kfold - test_score_efold)
    percentual_difference_percent = (percentual_difference / test_score_kfold) * 100
    print(f"Prozentuale Differenz zwischen k-fold und e-fold: {percentual_difference_percent:.2f}%")
    best_test_scores_diff[iteration] = percentual_difference_percent


