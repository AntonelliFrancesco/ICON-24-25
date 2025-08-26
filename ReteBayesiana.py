import pandas as pd
import pickle
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from pgmpy.estimators import HillClimbSearch, K2Score, BayesianEstimator
from pgmpy.models import BayesianNetwork
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from pgmpy.inference import VariableElimination
import seaborn as sns
import numpy as np

# Carica il dataset
def load_dataset(path):
    return pd.read_csv(path)

#seleziona le feature pre-release
def get_pre_release_features(df, target):
    feature = df.select_dtypes(include=['number']).columns
    post_release_feature = ['vote_average', 'vote_count', 'Box Office', 'Earnings', 'ROI', 'Oscar and Golden Globes awards', 'successo', 'IMDb score']  
    pre_release_feature = [col for col in feature if col not in post_release_feature and col != target]
    return pre_release_feature


# Selezione feature rilevanti
def top_feature(df, features, target):
    mi_scores = mutual_info_classif(df[features], df[target], discrete_features=False)
    mi_df = pd.DataFrame({'feature': features, 'MI': mi_scores})
    #mi_df_sorted = mi_df.sort_values('MI', ascending=False)
    
    # Grafico ordinato
    #plt.figure(figsize=(8, 6))
    #sns.barplot(data=mi_df_sorted, x='MI', y='feature', palette='viridis')
    #plt.title('Mutual Information - tutte le variabili')
    #plt.xlabel('MI score')
    #plt.ylabel('')
    
    # Etichette con i valori sopra le barre
    #for i, v in enumerate(mi_df_sorted['MI']):
    #   plt.text(v + 0.005, i, f"{v:.3f}", va='center')
    
    #plt.tight_layout()
    #plt.show()
    return mi_df.sort_values('MI', ascending=False).head(6)['feature'].tolist()


# Discretizzazione
def discretize_features(df, features):
    variabili_continue = ['Budget', 'popularity', 'Running time', 'Oscar and Golden Globes nominations', 'Actors Box Office %', 'Director Box Office %']
    discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    df[variabili_continue] = discretizer.fit_transform(df[variabili_continue])
    return df, discretizer


# Visualizzazione rete
def visualizeBayesianNetwork(model):
    G = nx.MultiDiGraph(model.edges())
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=350, node_color="#1c7dd3")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, arrowstyle="->", edge_color="purple")
    plt.title("BAYESIAN NETWORK GRAPH")
    plt.axis('off')
    plt.show()
    plt.close('all')


# CV della BN
def cross_validate_bn(df, target, model, selected_features, discretizer):
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    accs, precs, recalls, f1s, aucs = [], [], [], [], []
    train_errs, test_errs = [], []

    for fold, (train_idx, test_idx) in enumerate(rkf.split(df.drop(columns=[target]), df[target]), start=1):
        train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        variabili_continue = ['Budget', 'popularity', 'Running time', 'Oscar and Golden Globes nominations', 'Actors Box Office %', 'Director Box Office %']
        train_df[variabili_continue] = discretizer.transform(train_df[variabili_continue])
        test_df[variabili_continue] = discretizer.transform(test_df[variabili_continue])

        # Fit BN sul training
        infer = VariableElimination(model)

        # Predizione TRAIN
        y_true_train, y_pred_train = [], []
        for _, row in train_df.iterrows():
            q = infer.query(variables=[target], evidence={col: int(row[col]) for col in selected_features}, show_progress=False)
            states = q.state_names[target]
            pred = states[int(np.argmax(q.values))]
            y_true_train.append(int(row[target])); y_pred_train.append(int(pred))
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_err = 1 - train_acc

        # Predizione TEST
        y_true, y_pred, y_score = [], [], []
        for _, row in test_df.iterrows():
            q = infer.query(variables=[target], evidence={col: int(row[col]) for col in selected_features}, show_progress=False)
            states = q.state_names[target]
            probs = np.array(q.values)
            prob_pos = float(probs[states.index(1)])
            pred = states[int(np.argmax(probs))]
            y_true.append(int(row[target])); y_pred.append(int(pred)); y_score.append(prob_pos)

        test_acc = accuracy_score(y_true, y_pred)
        test_err = 1 - test_acc

        # Metriche
        accs.append(test_acc)
        precs.append(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        if len(set(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_score))
        train_errs.append(train_err)
        test_errs.append(test_err)

        #print(f"\nStatistiche sul Fold {fold}:")
        #print(f"TrainErr → Mean: {np.mean(train_errs):.5f} | Std: {np.std(train_errs):.5f} | Var: {np.var(train_errs):.5f}")
        #print(f"TestErr  → Mean: {np.mean(test_errs):.5f}  | Std: {np.std(test_errs):.5f} | Var: {np.var(test_errs):.5f}")
        #print(f"Accuracy → Mean: {np.mean(accs):.5f}       | Std: {np.std(accs):.5f} | Var: {np.var(accs):.5f}")

    # Riepilogo
    def stat(x): return np.mean(x), np.std(x), np.var(x)
    print("\n Medie ± std e varianza su tutti i fold:")
    print(f"Accuracy   : {stat(accs)[0]:.3f} ± {stat(accs)[1]:.3f} | Var: {stat(accs)[2]:.5f}")
    print(f"Precision  : {stat(precs)[0]:.3f} ± {stat(precs)[1]:.3f} | Var: {stat(precs)[2]:.5f}")
    print(f"Recall     : {stat(recalls)[0]:.3f} ± {stat(recalls)[1]:.3f} | Var: {stat(recalls)[2]:.5f}")
    print(f"F1-score   : {stat(f1s)[0]:.3f} ± {stat(f1s)[1]:.3f} | Var: {stat(f1s)[2]:.5f}")
    if aucs:
        print(f"ROC-AUC    : {stat(aucs)[0]:.3f} ± {stat(aucs)[1]:.3f} | Var: {stat(aucs)[2]:.5f}")
    print(f"Train Error: {stat(train_errs)[0]:.3f} ± {stat(train_errs)[1]:.3f} | Var: {stat(train_errs)[2]:.5f}")
    print(f"Test Error : {stat(test_errs)[0]:.3f} ± {stat(test_errs)[1]:.3f} | Var: {stat(test_errs)[2]:.5f}")

#  Costruzione rete
def bNetCreation(dataSet, target, selected_features):
    # Creiamo un nuovo DataFrame con le feature selezionate
    df = dataSet.copy()
    feature_data = df[selected_features]

    # Apprendiamo la struttura della rete utilizzando solo le feature selezionate 
    learned_structure = HillClimbSearch(feature_data).estimate(
        scoring_method=K2Score(feature_data),
    )

    # Aggiungiamo gli archi dalle feature al target (successo)
    additional_edges = [(col, target) for col in selected_features]  # Collega ogni feature al target
    model = BayesianNetwork(list(learned_structure.edges()) + additional_edges)

    # Fitting del modello con le feature selezionate + target
    model.fit(
         df[selected_features + [target]],  
        estimator=BayesianEstimator,
        prior_type='BDeu',
        n_jobs=-1,
        equivalent_sample_size=20 
    )

    # Verifica la validità del modello
    model.check_model()
    return model
#funzione che predice il successo di un film casuale
def predici_successo(discretizer, model, target):
    infer = VariableElimination(model)
    example = model.simulate(n_samples=1)
    ordered_cols = list(discretizer.feature_names_in_)
    for col in ordered_cols:
        print(f"  {col}: {example.iloc[0][col]}")
    example_array = pd.DataFrame(example.loc[[0], ordered_cols], columns=ordered_cols)
    discretized_example = discretizer.transform(example_array)[0]


    discretized_dict = {ordered_cols[i]: int(discretized_example[i]) for i in range(len(ordered_cols))}
    q = infer.query(variables=[target], evidence=discretized_dict, show_progress=False)
    state_names = q.state_names[target]
    probs = np.asarray(q.values, dtype=float)
    pred_label = state_names[int(np.argmax(probs))]
    prob_pos = float(probs[state_names.index(1)]) if 1 in state_names else None

    print(f"Predizione '{target}': {int(pred_label)}  |  P({target}=1) = {prob_pos:.3f}")
    return int(pred_label), prob_pos

def prob_successo_per_variabile(model, variabile, valori_variabile):
    # Dizionario interno con valori medi o realistici
    valori_fissi = {
        'Budget': 2.0,
        'popularity': 2.0,
        'Director Box Office %': 3.0,
        'Actors Box Office %': 1.0,
        'Oscar and Golden Globes nominations': 0.0,
        'Running time': 2.0
    }

    # Rimuovo la variabile che sto analizzando per non fissarla
    valori_fissi.pop(variabile, None)

    infer = VariableElimination(model)
    risultati = {}

    for valore in valori_variabile:
        evidenze = valori_fissi.copy()
        evidenze[variabile] = valore

        query = infer.query(variables=['successo'], evidence=evidenze, show_progress=False)
        prob_successo = query.values[1]  # indice 1 = successo=1
        risultati[valore] = round(prob_successo, 4)
    for valore, prob in risultati.items():
        print(f"{valore} → P(successo=1) = {prob}")
    return risultati

#stampa cpd
def visualizeCPD(model):
    cpd_list = model.get_cpds()
    for cpd in cpd_list:
        print(f"\nCPD per la variabile '{cpd.variable}':")
        print(cpd)

#salva il modello
def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

#carica il modello
def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


