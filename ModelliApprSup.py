import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt

def train_test_models(X, y):
    # Modelli da provare
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }

    # Griglie per la ricerca iperparametri
    param_grids = {
        "Random Forest": {
            'n_estimators': [200, 300],     
            'max_depth': [8, 10],             
            'min_samples_split': [5, 10],       
            'max_features': ['sqrt', 'log2']     
        },
        "Gradient Boosting": {
            'learning_rate': [0.05, 0.1],        
            'max_depth': [3, 4],                 
            'n_estimators': [200, 300],          
            'subsample': [0.8, 0.9],             
            'min_samples_split': [5, 10]         
        },
        "Decision Tree": {
            'criterion': ['gini', 'entropy'],    
            'max_depth': [4, 5, 6],              
            'min_samples_split': [2, 5, 10],     
            'ccp_alpha': [0.0, 0.0005, 0.001]    
        }
    }



    # Stratified K-Fold
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    # Addestramento e valutazione dei modelli
    for model_name, model in models.items():
        print(f"\n================ {model_name} =================")

        # Ricerca iperparametri e training
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=kf, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)  # Ricerca del miglior modello

        best_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-validation Score: {grid_search.best_score_:.3f}")

        # Valutazione del modello
        accuracies, f1_scores, precisions, recalls, roc_aucs, train_errors, test_errors = [], [], [], [], [], [], []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calcolo delle metriche
            acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            prec = precision_score(y_test, y_test_pred, average='weighted')
            rec = recall_score(y_test, y_test_pred, average='weighted')

            # ROC-AUC (solo se il modello ha la funzione predict_proba)
            if hasattr(best_model, "predict_proba"):
                fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
                roc_auc = auc(fpr, tpr)
            else:
                roc_auc = np.nan

            accuracies.append(acc)
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)
            roc_aucs.append(roc_auc)

            train_error = 1 - accuracy_score(y_train, y_train_pred)
            test_error = 1 - acc
            train_errors.append(train_error)
            test_errors.append(test_error)

        # Risultati aggregati
        def metric_summary(name, values):
            return f"{name}: {np.mean(values):.3f} ± {np.std(values):.3f} (var={np.var(values):.6f})"

        print("\nRisultati aggregati su 5 fold:")
        print(metric_summary("Accuracy", accuracies))
        print(metric_summary("Precision", precisions))
        print(metric_summary("Recall", recalls))
        print(metric_summary("F1-Score", f1_scores))
        if not all(np.isnan(roc_aucs)):
            print(metric_summary("ROC-AUC", [v for v in roc_aucs if not np.isnan(v)]))
        print(metric_summary("Train Error", train_errors))
        print(metric_summary("Test Error", test_errors))

        # Stampa il classification report solo alla fine (sui dati aggregati)
        y_pred = best_model.predict(X)  # Predizione finale su tutto il dataset
        print("\nClassification Report (Aggregato):\n", classification_report(y, y_pred, digits=3))
        print("Confusion Matrix (Aggregato):\n", confusion_matrix(y, y_pred))

