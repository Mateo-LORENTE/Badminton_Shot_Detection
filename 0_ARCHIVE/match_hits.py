import pandas as pd
import numpy as np
import pickle
import cv2

from api_mv import *

#Récupération des tags depuis l'API de métavidéo et correspondance avec le numéro de frame

def match_hits(videoId,fps,inputs_path,outputs_path):

    # Recharger le dataframe avec les tableaux d'origine depuis le fichier
    with open(outputs_path+'/'+videoId+'_df_predict.pkl', 'rb') as file:
        df_predict_with_players = pickle.load(file)

    matchs_report = get_matchs_report('df')

    match_id = matchs_report[matchs_report['videoId']==str(videoId)]['id'].iloc[0]

    tags_report = get_tags_report(match_id,'df')

    tags_report['timecode'] = tags_report['timecode'].astype('int')

    tags_report = tags_report.sort_values(by=['timecode'], ascending=True)

    tags_report = tags_report[tags_report['type']=='6']

    df_strokes = pd.DataFrame(columns = ['Frame','Hit'])

    for i, tags in tags_report.iterrows():
        frame_nb = round((tags['timecode']/1000)*fps)
        if tags['terrain_coordonnees']=="1":
            df_strokes.loc[len(df_strokes)] = [frame_nb,'top']
        elif tags['terrain_coordonnees']=="3":
            df_strokes.loc[len(df_strokes)] = [frame_nb,'bottom']
        # else:
        #     if tags['terrain_fond']!=None:
        #         df_strokes.loc[len(df_strokes)] = [frame_nb,'top']
        #     elif tags['terrain_avant']!=None:
        #         df_strokes.loc[len(df_strokes)] = [frame_nb,'bottom']
        
    df_strokes.set_index('Frame',inplace=True)
        
    df_predict_with_players['Hit'] = 'no'

    for i, row in df_predict_with_players.iterrows():
        try:
            hit = df_strokes.loc[i].values[0]
            df_predict_with_players.loc[i,'Hit'] = hit
        except:
            pass

     # Sauvegarder le dataframe avec les tableaux d'origine à l'aide de Pickle
    with open(outputs_path+'/'+videoId+'_df_predict_with_players_and_hits.pkl', 'wb') as file:
        pickle.dump(df_predict_with_players, file)

    return
    