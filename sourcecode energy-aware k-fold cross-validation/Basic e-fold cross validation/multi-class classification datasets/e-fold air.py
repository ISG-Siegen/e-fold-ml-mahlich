import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import t
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), np.std(data) / np.sqrt(n)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


# Daten importieren, ggf. pre processing
import pandas as pd

df = pd.read_csv('air_quality_health_impact_data.csv')
X = df.drop(['RecordID', 'HealthImpactClass'], axis=1)
y = df['HealthImpactClass']
k = 10

# Modelle in Liste definieren
models = [
    ("KNN", KNeighborsClassifier()),
    ("Logistic Regression", LogisticRegression()),
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("AdaBoost", AdaBoostClassifier(algorithm='SAMME'))
]

for model_name, model in models:
    print(f"Running model: {model_name}")
    all_fold_scores_all_runs = {}
    all_confidence_check = {}
    all_cancel_check = {}
    all_diff_check = {}

    f1_weighted_scorer = make_scorer(f1_score, average='weighted')

    for run in range(1, 101):
        random_state_value = run  # RandomState setzen (damit die Datenverteilung reproduziert werden kann, diese ändert sich in jeder schleife)
        kf = StratifiedKFold(n_splits=k, shuffle=True,
                             random_state=random_state_value)  # Hier wird definiert wie die Daten gesplittet werden sollen (StratifiedKFold gleichmäßige Klassenverhältniss in allen Folds)

        all_fold_scores = {}  # Die Scores der jeweiligen CV pro Fold
        all_folds_meanscore = {}  # Die Mean Scores der CV zum jeweiligen Fold
        std_after_each_fold = {}

        if model_name == 'Decision Tree':
            model = DecisionTreeClassifier(random_state=random_state_value)
        elif model_name == 'AdaBoost':
            model = AdaBoostClassifier(algorithm='SAMME', random_state=random_state_value)

        check = True
        std_count = 0

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train,
                      y_train)  # das Model wir dan nur auch den Train Daten trainiert (in dem Fall auf 9/10 der Daten (90%) weil k=10)
            model_score = f1_score(y_test, model.predict(
                X_test),
                                   average='weighted')  # hie rpassiert die evaluierung (f1-Score) nur auf den Test Daten (1/10) (10%)

            all_fold_scores[
                fold_idx] = model_score  # Der einzelne Scores der jeweiligen CV pro Fold wird mit dem Fold-Index (Fold nummerierung) und dem score in der oben definierten Liste gesepichert

            mean_scores = np.mean(list(
                all_fold_scores.values()))  # Mean Score aller aktuellen Folds (Durchläufe) berechnen (Als mean score über alle elemente die sich in der einzel Fold score Liste befinden
            all_folds_meanscore[
                fold_idx] = mean_scores  # und wird dann wieder mit dem Fold Index auch in der oben definierten Liste gespeichert

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

                    cancelation_fold = fold_idx
                    check = False

        if check == True:
            cancelation_fold = fold_idx
            print(f"Kein vorzeitiger Abbruch")

        all_fold_scores_all_runs[
            run] = all_folds_meanscore  # Hier werden alle Meanscores pro Fold zu dem jeweiligen run gespeichert
        scores = np.array(list(all_fold_scores.values()))
        mean_score, lower_bound, upper_bound = mean_confidence_interval(scores)

        print(f"Run:{run}")
        print(f"Werte pro Durchlauf:")
        print(all_fold_scores)
        print(f"Durchschnittswerte pro Durchlauf:")
        print(all_folds_meanscore)
        print("Standardabweichung der Scores nach jedem Fold")
        print(std_after_each_fold)
        #############

        # Daten für boxplot vorbereiten
        boxplot_data = []
        labels = []

        for i in range(1, fold_idx + 1):
            fold_scores = list(all_fold_scores.values())[:i]
            boxplot_data.append(fold_scores)
            labels.append(f'Fold {i}')

        fig, ax1 = plt.subplots(figsize=(14, 8))

        ax1.set_xlabel('Fold Number')
        ax1.set_ylabel('F1-Score')

        ax1.plot(list(all_fold_scores.keys()), list(all_fold_scores.values()), marker='o', color='b',
                 label='Single-Scores')
        ax1.plot(list(all_fold_scores.keys()), list(all_fold_scores.values()), linestyle='-', color='b')
        ax1.plot(list(all_folds_meanscore.keys()), list(all_folds_meanscore.values()), marker='o', color='g',
                 label='Mean-Scores')
        ax1.plot(list(all_folds_meanscore.keys()), list(all_folds_meanscore.values()), linestyle='-', color='g')
        ax1.axhline(y=lower_bound, color='orange', linestyle='--')
        ax1.axhline(y=mean_score, color='orange', linestyle='--', label='Confidence interval')
        ax1.axhline(y=upper_bound, color='orange', linestyle='--')
        if cancelation_fold:
            ax1.axvline(x=cancelation_fold, color='r',
                        linestyle='--')  # , #label=f'Abbruch bei Fold {cancelation_fold}')
            ax1.scatter(cancelation_fold, all_folds_meanscore[cancelation_fold], color='r', s=100, zorder=5,
                        label='Cancellation')

        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        # ax2.set_ylabel('Boxplot Werte')
        boxprops = dict(facecolor='blue', alpha=0.2)
        flierprops = dict(marker='o', markerfacecolor='white', markersize=8, linestyle='none')
        ax2.boxplot(boxplot_data, vert=True, patch_artist=True, labels=labels, widths=0.3, boxprops=boxprops,
                    flierprops=flierprops)

        fig.tight_layout()
        plt.title('F1-Scores Across Different Folds')
        plt.show()

        print(f"Tatsächlicher Wert nach 10 Fold: {all_folds_meanscore[10]}")
        print(f" Wert nach Abbruch bei Fold: {cancelation_fold} {all_folds_meanscore[cancelation_fold]}")

        all_cancel_check[run] = cancelation_fold
        filtered_values_all_cancel_check = [value for value in all_cancel_check.values() if value != 10]
        indices_with_value_10 = [key for key, value in all_cancel_check.items() if value == 10]

        diff_real_cancelation = abs(all_folds_meanscore[10] - all_folds_meanscore[cancelation_fold])
        diff_real_cancelation_change_percent = (diff_real_cancelation / all_folds_meanscore[10]) * 100

        all_diff_check[run] = diff_real_cancelation_change_percent
        filtered_values_all_diff_check = [value for key, value in all_diff_check.items() if
                                          key not in indices_with_value_10]

        if all_folds_meanscore[cancelation_fold] < upper_bound and all_folds_meanscore[cancelation_fold] > lower_bound:
            all_confidence_check[run] = True
        else:
            all_confidence_check[run] = False

        filtered_values_all_confidence_check = [value for key, value in all_confidence_check.items() if
                                                key not in indices_with_value_10]

    print("Evaluation")
    print("-------------------------------------")
    print("Cancellation within the confidence interval:")
    print("-------------------------------------")
    print("Check ob innerhalb des Konfidenzintervalls für alle 100 Iterationen :")
    print(all_confidence_check)
    count_true = np.array(list(all_confidence_check.values()).count(True))
    count_false = np.array(list(all_confidence_check.values()).count(False))
    print(f"Anzahl True: {count_true}  ,  Anzahl False: {count_false}")
    print("___---___")
    print("Check ob innerhalb des Konfidenzintervalls für alle Iterationen die nicht vorzeitig abgebrochen wurden :")
    print(filtered_values_all_confidence_check)
    count_true = np.array(list(filtered_values_all_confidence_check).count(True))
    count_false = np.array(list(filtered_values_all_confidence_check).count(False))
    print(f"Anzahl True: {count_true}  ,  Anzahl False: {count_false}")
    labels = ['True', 'False']
    counts = [count_true, count_false]
    plt.bar(labels, counts, color=['green', 'red'])
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.xlabel('')
    plt.ylabel('Count')
    plt.title('Cancellation within the confidence interval')
    plt.show()

    print("-------------------------------------")
    print("Early cancellation on fold:")
    print("-------------------------------------")

    print(f"Werte mit 10: {all_cancel_check}")
    print(f"Gefiltert ohne Wert 10: {filtered_values_all_cancel_check}")
    filtered_values_all_cancel_check_avg = np.mean(filtered_values_all_cancel_check)
    count_ten = np.array(list(all_cancel_check.values()).count(10))
    print(f"Abbruch nach AVG: {filtered_values_all_cancel_check_avg}   (Wert ohne die runs mit k=10)")
    print(f"Anzahl kein Abbruch: {count_ten} bei den Runs:{indices_with_value_10}")
    folds_with10 = list(all_cancel_check.values())
    # print(folds_with10)
    print("Abbruch nach Folds, alle 100 Iterationen")
    print(round(np.mean(folds_with10), 2))
    runs = list(all_cancel_check.keys())
    count_cancel = list(all_cancel_check.values())

    plt.bar(runs, count_cancel, color="blue")
    plt.axhline(y=filtered_values_all_cancel_check_avg, color='orange', linestyle='-', linewidth=2,
                label='Durchschnitt')
    plt.xlabel('Runs')
    plt.ylabel('Folds')
    plt.title('Early cancellation on fold:')
    plt.show()

    print("-------------------------------------")
    print("Percentage difference from the optimal value")
    print("-------------------------------------")
    print("Absoulte Differenz von e-fold zu 10-fold für alle 100 Iterationen")
    print(all_diff_check)
    print("___---___")
    print("Absoulte Differenz von e-fold zu 10-fold für alle Iterationen wo e<10 sind")
    print(filtered_values_all_diff_check)
    print("___---___")

    # avg_all_diff_check = np.mean(list(all_diff_check.values()))
    avg_all_diff_check = np.mean(filtered_values_all_diff_check)
    print(f"AVG Abweichung von tatsächlichen zum vorzeitigen Wert (ohne die k=10 Werte): {avg_all_diff_check}")
    print("gerundent:")
    print(round(avg_all_diff_check))
    # Diagramm:
    runsx = list(all_diff_check.keys())
    value = list(all_diff_check.values())

    plt.bar(runsx, value, color="silver")
    plt.axhline(y=avg_all_diff_check, color='orange', linestyle='-', linewidth=2, label='Durchschnitt')
    plt.xlabel('Runs')
    plt.ylabel('Difference')
    plt.title('Percentage difference from the "optimal" value')
    plt.show()

    print("-------------------------------------")
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

    method = '1'
    algo = model_name
    dataset = 'Air'

    # Speichern aller 100 Abbrüche
    json_data = json.dumps(all_cancel_check)
    cursor.execute('''
        INSERT INTO [dbo].[AvgAbbruchFold] (Method, Algortihm, Dataset, jsondata)
        VALUES (?, ?, ?, ?)
    ''', (method, algo, dataset, json_data))

    # Speichern Konfidenzintervall
    json_data = json.dumps(all_confidence_check)
    cursor.execute('''
        INSERT INTO [dbo].[Konfidenzintervall] (Method, Algortihm, Dataset, jsondata)
        VALUES (?, ?, ?, ?)
    ''', (method, algo, dataset, json_data))

    # Speichern aller % Abweichungen alle 100
    json_data = json.dumps(all_diff_check)
    cursor.execute('''
        INSERT INTO [dbo].[Prozent_Diff_ALL100] (Method, Algortihm, Dataset, jsondata)
        VALUES (?, ?, ?, ?)
    ''', (method, algo, dataset, json_data))

    # Speichern nur die %Abweichung die e<10 sind
    json_data = json.dumps(filtered_values_all_diff_check)
    cursor.execute('''
        INSERT INTO [dbo].[prozent_diff_smaller10] (Method, Algortihm, Dataset, jsondata)
        VALUES (?, ?, ?, ?)
    ''', (method, algo, dataset, json_data))

    conn.commit()

    # Verbindung schließen
    cursor.close()
    conn.close()
    print("DB saving complete")

    del all_fold_scores, all_folds_meanscore, std_after_each_fold, std_count, check, cancelation_fold