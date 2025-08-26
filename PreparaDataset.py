import pandas as pd
import ast



def prepara_dataset(path_df1, path_df2, output_path):
    # Funzione interna per pulire i generi
    def clean_genres(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            if not isinstance(genres, list):
                return []
            return genres
        except:
            return []

    # 1. Caricamento
    df1 = pd.read_csv(path_df1)
    df2 = pd.read_csv(path_df2, encoding='ISO-8859-1')

    # 2. Pulizia generi
    df1['genre_names'] = df1['genre_names'].apply(clean_genres)

    # 3. Colonne binarie per generi
    all_genres = set([genre for sublist in df1['genre_names'] for genre in sublist])
    for genre in all_genres:
        df1[genre] = df1['genre_names'].apply(lambda x: 1 if genre in x else 0)

    # 4. Drop colonne inutili
    df1 = df1.drop(columns=['id', 'overview', 'genre_names'])

    # 5. Pulizia premi
    df2['Oscar and Golden Globes awards'] = df2['Oscar and Golden Globes awards'].fillna(0).astype(int)

    # 6. Merge
    df_merged = pd.merge(df1, df2, how='inner', left_on='original_title', right_on='Movie')

    # 7. Drop colonne ridondanti
    df_merged = df_merged.drop(columns=[
        'Movie', 'Genre', 'Release year', 'Director',
        'Actor 1', 'Actor 2', 'Actor 3', 'release_date'
    ])

    # 8. ROI
    df_merged['ROI'] = (df_merged['Box Office']- df_merged['Budget'])/ df_merged['Budget']
    # 9. Percentuali
    df_merged['Actors Box Office %'] /= 100
    df_merged['Director Box Office %'] /= 100

    # Classificazione sulla copia
    df_merged['successo'] = ((df_merged['ROI'] >= 0.8) & (df_merged['IMDb score']>=7)).astype(int)

    # 13. Percentuale successi
    percentuale_successi = (df_merged['successo'].sum() / len(df_merged)) * 100
    print(f"Percentuale di successi: {percentuale_successi:.2f}%")

    # 14. Salvataggio
    df_merged.to_csv(output_path, index=False)
    print(f" Dataset unito e salvato come '{output_path}'")

    return df_merged
