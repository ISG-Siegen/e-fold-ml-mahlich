import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import t
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# Daten importieren, ggf. pre processing
from sklearn.datasets import load_breast_cancer

#Hyperparameter für e-fold cross-validation anpassen
e_max = 10
diff_max = 5
e_count = 2
e_check = 2


cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
k = e_max


model = KNeighborsClassifier()

random_state_value =29
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state_value)  # Hier wird definiert wie die Daten gesplittet werden sollen (StratifiedKFold gleichmäßige Klassenverhältniss in allen Folds)

all_fold_scores = {}  # Die Scores pro Fold
all_folds_meanscore = {}  # Die Mean Scores zum jeweiligen Fold
std_after_each_fold = {} #Std pro Fold

std_count = 0

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train,y_train)  # das Model wir da nur auf den Train Daten trainiert (in dem Fall auf 9/10 der Daten (90%) weil k=10)
    model_score = f1_score(y_test, model.predict(X_test))  # hie rpassiert die evaluierung (f1-Score) nur auf den Test Daten (1/10) (10%)

    all_fold_scores[fold_idx] = model_score  # Der einzelne Scores der jeweiligen CV pro Fold wird mit dem Fold-Index (Fold nummerierung) und dem score in der oben definierten Liste gesepichert

    mean_scores = np.mean(list(all_fold_scores.values()))  # Mean Score aller aktuellen Folds (Durchläufe) berechnen (Als mean score über alle elemente die sich in der einzel Fold score Liste befinden
    all_folds_meanscore[fold_idx] = mean_scores  # und wird dann wieder mit dem Fold Index auch in der oben definierten Liste gespeichert

    if fold_idx > 1:
    # Berechnung der Standardabweichung Scores nach jedem Fold
        std = np.std(list(all_fold_scores.values()),ddof=1)  # Hier std für Stichproben und nicht der Gesamtmenge
        std_after_each_fold[fold_idx] = std

    if fold_idx > e_check:
        actual_std = std  # aktuelle std
        last_std = std_after_each_fold[fold_idx - 1]  # vorherige std

        if actual_std < last_std:  # wenn die aktuelle kleiner ist als die vorherige erhöhen wir einen counter
            std_count += 1
        else:  # wenn nicht prüfen wir wie gravierend der Unterschied ist
            # Abweichung zum vorherigen gravierend ?
            std_change = abs(actual_std - last_std)
            std_change_percent = (std_change / actual_std) * 100
            if std_change_percent > diff_max:  # größer als 5% dann count zurücksetzen , kleiner dann erhöhen
                std_count = 0
            else:
                std_count += 1

    if std_count == e_count:  # Wenn 2 mal hintereinander geringere std und die nachfolgende ebenfalls dann aufhören
        print(f"abbruch nach {fold_idx} Folds")
        print(f"Model Performance liegt bei {mean_scores}")
        break


print(all_fold_scores)
print(all_folds_meanscore)
print(std_after_each_fold)