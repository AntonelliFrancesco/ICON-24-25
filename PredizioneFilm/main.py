from PreparaDataset import prepara_dataset
from ModelliApprSup import train_test_models
import pandas as pd
from ReteBayesiana import *

def main():
    target = "successo"

    # merge dei dataset e preparazione
    #dataset gia' creato
    #df_finale = prepara_dataset("movies2.csv", "movies_data.csv", "merged_dataset.csv")
    df_finale = load_dataset("merged_dataset.csv")

    # Selezione feature pre-release
    pre_release_feature = get_pre_release_features(df_finale, target)

    #  Subset e pulizia
    subset = df_finale[pre_release_feature + [target]].dropna().copy()
    
    X = subset[pre_release_feature]
    y = subset[target]

    train_test_models(X, y)
    
    # Top feature
    selected_features = top_feature(subset, pre_release_feature, target)
    print("Top feature selezionate:", selected_features)

    
    # Discretizzazione
    subset_discr, discretizer = discretize_features(subset.copy(), selected_features)

    # Creazione modello BN
    #model = bNetCreation(subset_discr, target, selected_features)
    #se il modello e' gia' stato creato
    model = load_model("BN.pkl")
    
    # Visualizzazione rete
    visualizeBayesianNetwork(model)

    # Cross-validation della bn
    cross_validate_bn(subset, target, model, selected_features, discretizer)

    # Predizione esempio casuale
    predici_successo(discretizer, model, target)
    prob_successo_per_variabile(model, "popularity", [0.0, 1.0, 2.0, 3.0])

    # Salvataggio modello
    save_model(model, "BN.pkl")
    visualizeCPD(model)

if __name__ == "__main__":
    main()
