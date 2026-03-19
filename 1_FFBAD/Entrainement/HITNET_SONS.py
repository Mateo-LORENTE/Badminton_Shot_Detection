import numpy as np
import glob
import os
import pickle
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Nouveaux pacakges dont on a besoin pour : Test_train_hitnet_V2
import tensorflow as tf
import math
import os
import pandas as pd
import numpy as np
import random
from scipy.stats import mode
from scipy.ndimage.interpolation import shift
from skimage.transform import rescale, resize

import shutil
#from camera_changes import *






import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import cv2





def extract_audio_features_per_frame(audio_path, video_path, sr=16000, n_mfcc=13, hop_duration=0.016):

    '''
    Extraction des features inspirés de extract_sons.
    2 types de features extraites :
        - MFCC features 1D (13 MFCC différentes par frame)
        - log mel-spectrogramme features 2D

    Args:
        audio_path (str): chemin vers l'audio (créé si absent).
        video_path (str): chemin vers la vidéo source.
        sr (int): taux d'échantillonnage souhaité .
        n_mfcc (int): nombre de MFCC .
        hop_duration (float): pas temporel pour features .

    Returns:
        np.ndarray: tenseur (N, 128, 6) des log-mel par frame.
        OU
        np.ndarray: tenseur (N, n_mfcc) des MFCC par frame.
    
    '''
        

    print(video_path)

    # Marqueur
    temp_audio_created = False

    if not os.path.exists(audio_path):
        local_video_path = video_path
        clip = VideoFileClip(local_video_path)
        clip.audio.write_audiofile(audio_path)
        temp_audio_created = True
    # Chargement audio et vidéo
    y, sr = librosa.load(audio_path, sr=None)
    y = y.astype(np.float32)
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Chargement des frames à analyser depuis le fichier .pkl
    pkl_path = f"/home/onyxia/work/DATA/pkl_merged_peak/{match_id}_df_merged_peak.pkl"
    print(pkl_path)
    df = pd.read_pickle(pkl_path)
    if 'Frame' not in df.columns:
        raise ValueError("La colonne 'Frame' est absente du fichier pkl")
    print(df.shape)
    frames = sorted(df['Frame'].unique())
    print(len(frames))
    # Paramètres MFCC
    n_mfcc = 13
    frame_duration = 1 / fps  # 1 frame vidéo = durée d'une frame audio
    n_fft = int(frame_duration * sr)
    hop_length = n_fft  # on garde 1 frame MFCC par fenêtre audio (sans chevauchement)

    # MFCC pour chaque frame
    mfcc_all = []
    for f in frames:
        start_sec = f / fps
        start_sample = int(start_sec * sr)
        end_sample = int((start_sec + 1 / fps) * sr)
        y_frame = y[start_sample:end_sample]

        
        #mfcc = librosa.feature.mfcc(y=y_frame, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
        #mfcc_all.append(mfcc.T[0])  # On prend le seul vecteur MFCC généré
        
            # ➤ Calcul du log-Mel spectrogramme
        S = librosa.feature.melspectrogram(
            y=y_frame,
            sr=sr,
            n_fft=1024,         # ou ton n_fft si défini
            hop_length=256,     # ou ton hop_length
            n_mels=128,
            fmin=50
        )
        log_mel = librosa.power_to_db(S, ref=np.max)
        target_T = 6  # ou 7 si tu préfères

        # Transpose pour avoir (T, 128) puis reshape
        log_mel = log_mel.astype(np.float32)

        if log_mel.shape[1] < target_T:
            # ➕ Padding en répétant la dernière colonne
            last_col = log_mel[:, -1][:, np.newaxis]
            pad = np.repeat(last_col, target_T - log_mel.shape[1], axis=1)
            log_mel = np.concatenate([log_mel, pad], axis=1)

        elif log_mel.shape[1] > target_T:
            # ✂️ Tronquer les colonnes en trop
            log_mel = log_mel[:, :target_T]

        # Append à la liste
        mfcc_all.append(log_mel)

    mfcc_final = np.stack(mfcc_all, axis=0)  
    print("MFCC shape final :", mfcc_final.shape)
    # Nettoyage
    if temp_audio_created:
        try:
            os.remove(audio_path)
            print(f"🔁 Audio temporaire supprimé : {audio_path}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la suppression : {e}")
    
    if fps < 28 :
        decalage = 0 # nombre de frames après le 


    elif match_id == "43835"  or match_id == "63851":
        decalage = 0

    elif match_id == "43695" or match_id == "47563" or match_id == "49887" :
        decalage = 1

    elif match_id == "46959" or match_id == "49766 ":
        decalage = -1

    else: 
        decalage = 6 

    start = 11 - decalage

    if decalage == 0 :
        end = None
        mfcc_cut = mfcc_final[start:end]
    
    if decalage < 0 :
        end = None
        mfcc_cut = mfcc_final[start:end]
        last_vec = mfcc_cut[-1]
        last_vec = last_vec[np.newaxis, :, :]  # devient (1, 128, 6)
        for _ in range(abs(decalage)):
            mfcc_cut = np.vstack([mfcc_cut, last_vec])
        
    if decalage > 0:
        end = -decalage
        mfcc_cut = mfcc_final[start:end]


    return mfcc_cut




def creer_mfcc_temporel_cnn2d(mfcc_array, num_consec=12):
    """
    Génère des séquences temporelles de spectrogrammes 2D centrées.

    Args:
        mfcc_array (np.ndarray): shape (N, H, W)
        num_consec (int): nombre de spectres consécutifs par séquence

    Returns:
        np.ndarray: shape (N, num_consec, H, W, C)
    """
    N, H, W = mfcc_array.shape
    sequences = []

    for i in range(N):
        start = i 
        end = i + num_consec

        if start < 0:
            pad = np.repeat(mfcc_array[0:1], repeats=-start, axis=0)
            seq = np.concatenate([pad, mfcc_array[0:end]], axis=0)
        elif end > N:
            pad = np.repeat(mfcc_array[-1:], repeats=end - N, axis=0)
            seq = np.concatenate([mfcc_array[start:], pad], axis=0)
        else:
            seq = mfcc_array[start:end]

        sequences.append(seq)

    return np.stack(sequences)  # (N, num_consec, H, W, C)



def creer_mfcc_temporel_centre(mfcc_array, num_consec=12):
    """
    Génère un tableau (N, num_consec, 13) avec des fenêtres centrées.
    Ajoute du padding (copie de première ou dernière frame) pour conserver N séquences.

    Args:
        mfcc_array (np.ndarray): tableau de shape (N, 13)
        num_consec (int): longueur des séquences (doit être pair)

    Returns:
        np.ndarray: tableau de shape (N, num_consec, 13)
    """
    assert num_consec % 2 == 0, "num_consec doit être pair pour un centrage symétrique."
    
    N, D = mfcc_array.shape
    half = num_consec // 2
    sequences = []

    for i in range(N):
        start = i - half
        end = i + half

        if start < 0:
            pad = np.repeat(mfcc_array[0:1], repeats=-start, axis=0)
            seq = np.concatenate([pad, mfcc_array[0:end]], axis=0)
        elif end > N:
            pad = np.repeat(mfcc_array[-1:], repeats=end - N, axis=0)
            seq = np.concatenate([mfcc_array[start:], pad], axis=0)
        else:
            seq = mfcc_array[start:end]

        sequences.append(seq)

    return np.stack(sequences)



#on définit une fonction qui va nous permettre de charger nos données
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pre_processing_Hitnet_Que_Volant_V3(folder_path):
    """
    Prétraite les données de matchs pour préparer un ensemble d'entraînement et un ensemble de test.

    Cette fonction :
      - Charge tous les fichiers .pkl du dossier donné.
      - Nettoie et renomme les colonnes pour normaliser les coordonnées du volant.
      - Sépare les données en train/test (70% / 30%).
      - Interpole et rééchantillonne les trajectoires du volant.
      - Encode les labels ('hit' et 'Peak_binary').
      - Extrait des caractéristiques temporelles (fenêtres glissantes) des coordonnées.
      - Extrait les features audio associées via `extract_audio_features_per_frame`.
      - Mise à l'échelle des features et création des jeux `x_train`, `y_train`, `x_test`, `y_test`.
      - Applique un sous-échantillonnage pour équilibrer les classes.

    Args:
        folder_path (str): chemin vers le dossier contenant les fichiers .pkl de données.

    Returns:
        tuple: (x_train, y_train_encoded, peak_train, mfcc_train,
                x_test, y_test_encoded, peak_test_encoded, mfcc_test)
            - x_train / x_test : caractéristiques trajectoire normalisées.
            - y_train_encoded / y_test_encoded : labels encodés (0: no, 1: bottom, 2: top).
            - peak_train / peak_test_encoded : labels de détection de pics binaires.
            - mfcc_train / mfcc_test : tenseurs audio alignés par frame.
    """




    # Obtention de la liste des fichiers .pkl dans le dossier
    file_list = glob.glob(os.path.join(folder_path, '*.pkl'))

    print("file_list de pre_processing : " ,file_list)

    # Liste pour stocker les données chargées à partir des fichiers .pkl
    data_list = []

    # Chargement des fichiers .pkl et concaténation des données



    for file in file_list:  ##TESTRAPIDE
        data = unpickle(file)
        data['hit'] = data['hit'].astype(str)

        data_list.append(data)

    #print(data_list)

    for idx, df in enumerate(data_list):
        oth = set(df['hit'].unique()) - {'no','bottom','top'}
        if oth:
            print(f"Fichier #{idx} a des hits inattendus : {oth}")

    # Compter les occurrences des différentes valeurs 
    unique_values, counts = np.unique(data_list[0]['hit'], return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"Valeur {value}: {count} occurrences")



    # # Renommons les colonnes pour les 4 dataframe
    for df in data_list:
        df.rename(columns={
            'shuttle_x': 'norm_shuttle_x',
            'shuttle_y': 'norm_shuttle_y',
            'ShuttleX': 'norm_shuttle_x',
            'ShuttleY': 'norm_shuttle_y',
            # 'hit' reste 'hit', pas besoin de le renommer
        }, inplace=True)


    

    # Maintenant ces colonnes existent, vous pouvez filtrer :
    for i in range(len(data_list)):
        data_list[i] = data_list[i][['norm_shuttle_y','norm_shuttle_x','hit','Peak_binary']].copy()
        
    #####################################################################################################################################
    ####################### On fait deux listes une pour les matchs d'entrainement et une pour les matchs de test #######################
    #####################################################################################################################################
 
    
    
    print("#####################################")
    print("## On fait deux listes une pour les matchs d'entrainement et une pour les matchs de test ##")
    print("#####################################")
    
    total_elements = len(data_list)
    indice_separation = round(total_elements * 0.3) # On arrondi à l'entier le plus proche

    # Séparation en deux sous-listes
    data_list_test = data_list[:indice_separation]
    data_list_train = data_list[indice_separation:]

    
    
    # # Ajoutez immédiatement :  MODIF MATEO 
    for df in data_list_train + data_list_test:
        df['hit'] = df['hit'].astype(str)

    print("liste des match d'entrainement : ", len(data_list_train))
    print("liste des match de test : ", len(data_list_test))

    

    # Compter les occurrences des différentes valeurs 
    unique_values, counts = np.unique(data_list_train[0]['hit'], return_counts=True)

    for value, count in zip(unique_values, counts):
        print(f"Valeur pour les train {value}: {count} occurrences")

    # Compter les occurrences des différentes valeurs 
    unique_values1, counts1 = np.unique(data_list_test[0]['hit'], return_counts=True)

    for value, count in zip(unique_values1, counts1):
        print(f"Valeur pour les test {value}: {count} occurrences")


        

    ####################################################################################################
    ################################### Pour les données d'entrainement ################################
    ####################################################################################################

    
    print("#####################################")
    print("## Pour les données d'entrainement ##")
    print("#####################################")
    
    ### Pour le volant
    # Liste pour stocker les données
    df_volant = []

    for i in range(len(data_list_train)):
        # Sélection des colonnes avec le mot "bottom" mais pas le mot "court" pour prendre que les coordonnnées des joueurs
        volant_cols = [col for col in data_list_train[i].columns if 'norm_shuttle' in col]
        # Crééons un DataFrame à partir des colonnes sélectionnées
        df_volant_i = data_list_train[i][volant_cols]
        
        # Différents calculs en fonction de si c'est des x ou des y
        x_cols = [col for col in df_volant_i.columns if '_x' in col]
        y_cols = [col for col in df_volant_i.columns if '_y' in col]

       
        # Ajout du DataFrame transformé à la liste df_bottom
        df_volant.append(df_volant_i)


    ##############  On regroupe les x et y ==> en vérité on n'utilisera pas les données étant passé dans la fonction combine_columns dans la suite
    # Fonction pour regroupe les colonnes par paire de x et y
    def combine_columns(row):
        x = row[:-1:2]  # Sélection des colonnes avec les indices pairs (x)
        y = row[1::2]   # Sélection des colonnes avec les indices impairs (y)
        return [list(a) for a in zip(x, y)]
    
    ######### Pour le volant
    for i in range(len(data_list_train)):
        # On applique la fonction à chaque ligne du DataFrame
        df_volant[i] = df_volant[i].apply(combine_columns, axis=1)

    ####################################################################################################
    ################################### On fait la même pour les données de test #######################
    ####################################################################################################
    
    
    print("#####################################")
    print("## On fait la même pour les données de test ##")
    print("#####################################")

    ### Pour le volant
    # Liste pour stocker les données
    df_volant_test = []

    for i in range(len(data_list_test)):
        # Sélection des colonnes avec le mot "bottom" mais pas le mot "court" pour prendre que les coordonnnées des joueurs
        volant_cols = [col for col in data_list_test[i].columns if 'norm_shuttle' in col]
        # Crééons un DataFrame à partir des colonnes sélectionnées
        df_volant_i = data_list_test[i][volant_cols]
        
        # Différents calculs en fonction de si c'est des x ou des y
        x_cols = [col for col in df_volant_i.columns if '_x' in col]
        y_cols = [col for col in df_volant_i.columns if '_y' in col]
    
        
        # Ajout du DataFrame transformé à la liste df_bottom
        df_volant_test.append(df_volant_i)

    ##############  On regroupe les x et y ==> en vérité on n'utilisera pas les données étant passé dans la fonction combine_columns dans la suite
    ######### Pour le volant
    for i in range(len(data_list_test)):
        # On applique la fonction à chaque ligne du DataFrame
        df_volant_test[i] = df_volant_test[i].apply(combine_columns, axis=1)

    #### Début du traitement sur les données d'entrainements
    def scale_data(x):
        # Convertir x en un tableau numpy
        x = np.array(x)
        cols = [i for i in range(x.shape[1])]
        # Fonction interne pour mettre à l'échelle par colonne
        def scale_by_col(x, cols):
            # Extraire les colonnes spécifiées
            x_ = np.array(x[:, cols])
            
            # Trouver les indices où les valeurs sont proches de zéro (éviter la division par zéro)
            idx = np.abs(x_) < eps
            
            # Trouver les valeurs minimales et maximales en excluant les valeurs proches de zéro
            m, M = np.min(x_[~idx]), np.max(x_[~idx])
            
            # Mettre à l'échelle les valeurs en utilisant la formule de mise à l'échelle min-max
            x_[~idx] = (x_[~idx] - m) / (M - m) + 1
            
            # Mettre à jour les colonnes d'origine avec les valeurs mises à l'échelle
            x[:, cols] = x_
            
            return x
        
        
        # Appeler la fonction interne pour mettre à l'échelle les colonnes spécifiées
        x_scaled = scale_by_col(x, cols)
        
        return x_scaled 


    class Trajectory(object):
        def __init__(self, df_name, interp=True):

            # df avec que les positions
            trajectory = df_name[['norm_shuttle_x', 'norm_shuttle_y']] 
            # On renomme
            trajectory = trajectory.rename(columns={'norm_shuttle_x': 'X', 'norm_shuttle_y': 'Y'})
            if interp:
                
                # On remplace les zéros par des NaN dans la colonne pour pouvoir utiliser l'interpolation
                trajectory['Y'] = trajectory['Y'].replace(0, np.nan)
                trajectory['X'] = trajectory['X'].replace(0, np.nan)

                ########### Pour Y
                # Liste des masques pour chaque cas d'interpolation en fonction du nombre de NaN consécutifs
                masksy = []
                # i va prendre de 1 à 5
                for i in range(1, 6):
                    # mask de 5 NaN de suite mais entouré de valeurs ok
                    mask = (trajectory['Y'].isna() &
                            trajectory['Y'].shift(-i).isna() &
                            trajectory['Y'].shift(-6).notna() &
                            (trajectory['Y'].shift(1).notna() if i == 1 else True))
                    masksy.append(mask)

                # interpolation pour chaque cas 
                for i, mask in enumerate(masksy, start=1):
                    # Pour toutes les lignes où on est dans ce cas on fait l'interpolation linéaire 
                    rowsY = trajectory.loc[mask].index
                    for row in rowsY:
                        trajectory.loc[row-1:row+i, 'Y'] = (
                            trajectory.loc[row-1:row+i, 'Y']
                            .interpolate(method='linear', limit=i, limit_direction='forward')
                        )
                ########### Idem pour X
                masksx = []
                for i in range(1, 6):
                    mask = (trajectory['X'].isna() &
                            trajectory['X'].shift(-i).isna() &
                            trajectory['X'].shift(-6).notna() &
                            (trajectory['X'].shift(1).notna() if i == 1 else True))
                    masksx.append(mask)

                for i, mask in enumerate(masksx, start=1):
                    rowsY = trajectory.loc[mask].index
                    for row in rowsY:
                        trajectory.loc[row-1:row+i, 'X'] = (
                            trajectory.loc[row-1:row+i, 'X']
                            .interpolate(method='linear', limit=i, limit_direction='forward')
                        )

                # on remet les 0 à la place des NaN quand il n'y a pas eu l'interpolataion
                trajectory['Y'] = trajectory['Y'].replace(np.nan, 0)
                trajectory['X'] = trajectory['X'].replace(np.nan, 0)
                #####################################################
                # listes X et Y interpolées
                Xb, Yb = trajectory.X.tolist(), trajectory.Y.tolist()
            else:
                # Si l'interpolation n'est pas activée, nous utilisons simplement les valeurs de X et Y du fichier. => faux on va tjrs l'utiliser
                Xb, Yb = trajectory.X.tolist(), trajectory.Y.tolist()


            # Stockage des trajectoires X et Y
            self.X = Xb
            self.Y = Yb

    identity = lambda x: x

    def resample(series, s):
        flatten = False
        
        # Vérifier si les données sont un tableau 1D, sinon les remodeler en 2D
        if len(series.shape) == 1:
            series.resize((series.shape[0], 1))  # Remodelage en un tableau 2D
            series = np.array(series)  # Conversion en un tableau numpy
            series = series.astype('float64')  # Conversion en type de données float64
            flatten = True
        
        if np.all(np.isnan(series)):
            series = np.zeros_like(series)
        

        # Rééchantillonnage des données avec une nouvelle taille en fonction de 's'
        series = resize(
            series, (int(s * series.shape[0]), series.shape[1]),
        )
        
        # Si les données ont été aplaties, les remodeler en 1D
        if flatten:
            series = series.flatten()
        
        return series
    


    ###########################
    # Définition du nombre de trames consécutives pour les fenêtres glissantes
    num_consec = 12  # Jui: Lire le document pour les paramètres corrects

    #Définissons la valeur epsilon pour éviter la division par zéro dans la fonction scale_data
    eps = 1e-8

    # Taille de la fenêtre à gauche et à droite pour le traitement des données
    left_window = 6
    right_window = 0

    ################################################################## Pour l'entrainement
    
    print("#####################################")
    print("## On passe au vrai traitement pour les données d'entrainements ##")
    print("#####################################")

    # Liste des noms de matchs
    matches = list('df_train_numero' + str(i) + '.pkl' for i in range(1,len(data_list_train)+1))

    # Listes vides pour stocker les données d'entraînement
    x_train, y_train, peaks_train, mfcc_train = [], [], [], []

    # Parcours de chaque match dans la liste "matches"
    #for match in matches:
    for k in range(len(data_list_train)):
        print(matches[k])
        print(data_list_train[k].shape)
        df = pd.read_pickle(file_list[k+len(data_list_test)])
        print(df.shape)

        # Parcours de différentes vitesses (ici, seule la vitesse 1.0 est utilisée)
        for speed in [1.0]:  # Liste des vitesses à considérer => traitement différent en fonction de la vitesse 
            
            
            # Lecture de la vidéo pour obtenir les dimensions (hauteur et largeur) des images
            height, width = 1080, 1920
            


            mapping = {'no': 0, 'bottom': 1, 'top': 2}

            df_hit = pd.DataFrame({'hit': data_list_train[k]['hit']})
            
            hit     = df_hit['hit'].map(mapping)
            print(type(hit))
             
            print("Nombre d'occurence après le mapping")
            unique_values, counts = np.unique(hit, return_counts=True)
            for value, count in zip(unique_values, counts):
                print(f"Valeur pour les train {value}: {count} occurrences")

            # On converti en dataframe
            df_hit= pd.DataFrame(data=hit)
            # On accède au valeur sous forme de série
            hit = df_hit.values[:, 0]
            


            mapping2 = {0: 0, 1: 1}

            df_peak = pd.DataFrame({'Peak_binary': data_list_train[k]['Peak_binary']})
            peak= df_peak['Peak_binary'].map(mapping2)

            # On converti en dataframe
            df_peak= pd.DataFrame(data=peak)
            # On accède au valeur sous forme de série
            peak = df_peak.values[:, 0]


            # Mais speed vaut 1
            if speed < 1:
                # NTS: speed < 1 signifie en fait que la séquence devient plus rapide
                hit = hit + shift(hit, -1) + shift(hit, +1)
                
            trajectory = Trajectory(data_list_train[k], interp=True)
            # Accès aux attributs Xb : trajectory.X et Yb : trajectory.Y
            
            # Rééchantillonnage des coordonnées X et Y de la trajectoire
            trajectory.X = resample(np.array(trajectory.X), speed)
            trajectory.Y = resample(np.array(trajectory.Y), speed)
            
            # Rééchantillonnage des données de frappe ("hit")
            hit = resample(hit, speed).round()
            peak = resample(peak, speed).round()

            
            
            # Création d'un tableau numpy pour les nouvelles valeurs de frappe
            y_new = np.array(hit)
            peak_new = np.array(peak)

            # Initialisation des listes pour les données d'entraînement
            x_list, p_list, y_list = [], [], []

            # Parcourir les séquences de données
            for i in range(num_consec):
                end = min(len(trajectory.X), len(hit)) - num_consec + i + 1
                #print("end :", end)

                # Création des caractéristiques x en utilisant les coordonnées de la trajectoire et des poses
                x_bird = np.array(list(zip(trajectory.X[i:end], trajectory.Y[i:end])))
                
                x = np.hstack([x_bird])
                print("x shape", x.shape)

                # Étiquettes y pour la séquence actuelle
                y = y_new[i:end]
                p_seq = peak_new[i:end]
                # Ajout des données d'entraînement à x_list et y_list
                x_list.append(x)
                y_list.append(y)
                p_list.append(p_seq)

            # Empilement horizontal de toutes les données d'entraînement x
            x_t = np.hstack(x_list)

            # Création des étiquettes y en utilisant la fenêtre gauche et éventuellement la fenêtre droite
            if right_window > 0:
                y_t = np.max(np.column_stack(y_list[left_window:-right_window]), axis=1)
                p_t = np.max(np.column_stack(p_list[left_window:-right_window]), axis=1)
            else:
                y_t = np.max(np.column_stack(y_list[left_window:]), axis=1)
                p_t = np.max(np.column_stack(p_list[left_window:]), axis=1)
            video_filename = os.path.basename(file_list[k+len(data_list_test)]).split('_')[0]
            video_path = f"s3/lorentemateo/{video_filename}.mp4"
            audio_path = f"/home/onyxia/work/DATA/sons/{video_filename}.wav"

            mfcc_t = extract_audio_features_per_frame(audio_path=audio_path,video_path=video_path)
            print(f"mfcc_shape : {mfcc_t.shape}")
            # Liste des fonctions d'augmentation des données à appliquer
            augmentations = [identity]  

            # Parcourir chaque fonction d'augmentation et appliquer aux données d'entraînement
            for transform in augmentations:
                # Appliquer la transformation à x_t (caractéristiques)
                #print("x_t avant", x_t)
                transformed_x = transform(x_t)
                print("x_t première ligne", len(transformed_x[0]))
                print("transformed_x shape[0]", transformed_x.shape[0])
                print("transformed_x shape[1]", transformed_x.shape[1])

                x_train.append(scale_data(transformed_x))
                print(scale_data(transformed_x).shape)
                #print("x_train après scale_data:", x_train)
                y_train.append(y_t)
                print(y_t.shape)
                peaks_train.append(p_t)
                print(mfcc_t.shape)
                #mfcc_train.append(creer_mfcc_temporel_centre(mfcc_t)
                # mfcc_t = mfcc_t[..., np.newaxis]
                # mfcc_train.append(creer_mfcc_temporel_cnn2d(mfcc_t))
                mfcc_train.append(mfcc_t)

    print("Voilà à quoi ressemble x_train[0] AVANT l'empliement verticale", x_train[0]) 
    print("Voilà à quoi ressemble x_train AVANT l'empliement verticale", x_train) 
    # Empilement vertical de toutes les données d'entraînement x après les transformations
    x_train = np.vstack(x_train)
    print("Voilà à quoi ressemble x_train APRES l'empliement verticale", x_train) 

    print("#######################################################")
    print("##################### Pour y ##########################")
    print("#######################################################")

    print("Voilà à quoi ressemble y_train AVANT l'empliement verticale", y_train) 
    # Empilement horizontal de toutes les étiquettes y d'entraînement
    y_train = np.hstack(y_train)
    print("Voilà à quoi ressemble y_train APRES l'empliement verticale", y_train)
    peaks_train = np.hstack(peaks_train)
    mfcc_train = np.vstack(mfcc_train)
    print(mfcc_train.shape)
    print(x_train.shape)

    ####################################################################################################################################
    ################################################################## Pour le test ####################################################
    ####################################################################################################################################
    
    print("#####################################")
    print("## On passe au vrai traitement pour les données de test ##")
    print("#####################################")

    # Liste des noms de matchs
    matches_test = list('df_test_numero' + str(i) + '.pkl' for i in range(1,len(data_list_test)+1))

    # Listes vides pour stocker les données d'entraînement
    x_val, y_val, peaks_test, mfcc_test = [], [], [], []

    # Parcours de chaque match dans la liste "matches"
    #for match in matches:
    for k in range(len(data_list_test)):
        
        # Parcours de différentes vitesses (ici, seule la vitesse 1.0 est utilisée)
        for speed in [1.0]:  # Liste des vitesses à considérer => traitement différent en fonction de la vitesse 
        
            # Création de listes vides pour stocker les données
            x_list, y_list = [], []
            
            # Lecture de la vidéo pour obtenir les dimensions (hauteur et largeur) des images
            height, width = 1080, 1920

            mapping = {'no': 0, 'bottom': 1, 'top': 2}
            df_hit = pd.DataFrame({'hit': data_list_test[k]['hit']})
            hit     = df_hit['hit'].map(mapping)
            # On utilise map() pour encoder la colonne 'hit' avec les valeurs du dictionnaire
            
            # On converti en dataframe
            # On accède au valeur sous forme de série
            # On converti en dataframe
            df_hit= pd.DataFrame(data=hit)
            hit = df_hit.values[:, 0]
            #hit = hit.values[:, 1]
            # Mais speed vaut 1
            if speed < 1:
                # NTS: speed < 1 signifie en fait que la séquence devient plus rapide
                hit = hit + shift(hit, -1) + shift(hit, +1)
                
            trajectory = Trajectory(data_list_test[k], interp=True)
            # Accès aux attributs Xb : trajectory.X et Yb : trajectory.Y

            mapping2 = {0: 0, 1: 1}

            df_peak = pd.DataFrame({'Peak_binary': data_list_test[k]['Peak_binary']})
            peak= df_peak['Peak_binary'].map(mapping2)

            # On converti en dataframe
            df_peak= pd.DataFrame(data=peak)
            # On accède au valeur sous forme de série
            peak = df_peak.values[:, 0]
            
            # Rééchantillonnage des coordonnées X et Y de la trajectoire
            trajectory.X = resample(np.array(trajectory.X), speed)
            trajectory.Y = resample(np.array(trajectory.Y), speed)
            
            # Rééchantillonnage des données de frappe ("hit")
            hit = resample(hit, speed).round()
            peak = resample(peak, speed).round()
            
            # Création d'un tableau numpy pour les nouvelles valeurs de frappe
            y_new = np.array(hit)
            peak_new = np.array(peak)

            # Initialisation des listes pour les données d'entraînement
            x_list, p_list, y_list = [], [], []

            # Parcourir les séquences de données
            for i in range(num_consec):
                end = min(len(trajectory.X), len(hit)) - num_consec + i + 1
                #print("end :", end)

                # Création des caractéristiques x en utilisant les coordonnées de la trajectoire et des poses
                x_bird = np.array(list(zip(trajectory.X[i:end], trajectory.Y[i:end])))
                
                x = np.hstack([x_bird])
                print("x shape", x.shape)

                # Étiquettes y pour la séquence actuelle
                y = y_new[i:end]
                p_seq = peak_new[i:end]
                # Ajout des données d'entraînement à x_list et y_list
                x_list.append(x)
                y_list.append(y)
                p_list.append(p_seq)

            # Empilement horizontal de toutes les données d'entraînement x
            x_t = np.hstack(x_list)

            # Création des étiquettes y en utilisant la fenêtre gauche et éventuellement la fenêtre droite
            if right_window > 0:
                y_t = np.max(np.column_stack(y_list[left_window:-right_window]), axis=1)
                p_t = np.max(np.column_stack(p_list[left_window:-right_window]), axis=1)
            else:
                y_t = np.max(np.column_stack(y_list[left_window:]), axis=1)
                p_t = np.max(np.column_stack(p_list[left_window:]), axis=1)

            video_filename = os.path.basename(file_list[k]).split('_')[0]
            video_path = f"s3/lorentemateo/{video_filename}.mp4"
            audio_path = f"/home/onyxia/work/DATA/sons/{video_filename}.wav"

            mfcc_t = extract_audio_features_per_frame(audio_path=audio_path,video_path=video_path)
            print(mfcc_t.shape)

            x_val.append(scale_data(x_t))
            y_val.append(y_t)
            peaks_test.append(p_t)
            #mfcc_test.append(creer_mfcc_temporel_centre(mfcc_t))
            # mfcc_t = mfcc_t[..., np.newaxis]
            # mfcc_test.append(creer_mfcc_temporel_cnn2d(mfcc_t))
            mfcc_test.append(mfcc_t)
            

    x_val = np.vstack(x_val)
    y_val = np.hstack(y_val)
    peaks_test = np.hstack(peaks_test)
    mfcc_test = np.vstack(mfcc_test)

    
    print("#####################################")
    print("## Première étape finitooooooo : on a x_train, y_train, x_val et y_val ##")
    print("#####################################")

    print(" On a donc :")
    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    print("x_val",x_val.shape)
    print("y_val",y_val.shape)
    print(mfcc_test.shape)
    
    print("#####################################")
    print("## On va sous échantillonner les données d'entrainement ##")
    print("#####################################")
    
    ### Pour les données d'entrainement
    # Calcul du rapport entre les valeurs positives (frappe de l'un de joueur) et nulles (pas de frappe) dans y_train
    p_train = sum(y_train > 0) / sum(y_train == 0) # =0.24295678603436854 environ
    print("Le nombre de y supérieur à 0",sum(y_train > 0))
    print("Le nombre de y égale à 0",sum(y_train == 0))
    print("la valeur de p est :", p_train)

    # Création d'une liste vide pour stocker l'échantillon subsample
    subsample = []

    ############# Parcours de chaque ligne de x_train
    for i in range(x_train.shape[0]):
        # Si un nombre aléatoire entre 0 et 1 est inférieur à p ou si y_train[i] est positif,
        # on ajoute l'indice i à la liste subsample
        if random.random() < p_train or y_train[i]:
            subsample.append(i)

    print("la longueur de subsample est de : ", len(subsample))

    x_train = x_train[subsample]
    y_train_encoded = y_train[subsample]
    peak_train = peaks_train[subsample]
    mfcc_train = mfcc_train[subsample]

    # SI ON VEUT LE DATAFRAME ENTIER ECHNAGER AVEC CI-DESSOUS

    # x_train = x_train
    # y_train_encoded = y_train
    # peak_train = peaks_train
    # mfcc_train = mfcc_train

    print("#####################################")
    print("## On va sous échantillonner les données de test ##")
    print("#####################################") 
    
    ############# Pour les données de test
    p_test = sum(y_val > 0) / sum(y_val == 0)
    subsample = []
    for i in range(x_val.shape[0]):
        if random.random() < p_test or y_val[i]:
            subsample.append(i)
            
    x_test = x_val[subsample]
    y_test_encoded = y_val[subsample] 
    peak_test_encoded = peaks_test[subsample]
    mfcc_test = mfcc_test[subsample]

    # SI ON VEUT LE DATAFRAME ENTIER ECHNAGER AVEC CI-DESSOUS

    # x_test = x_val
    # y_test_encoded = y_val
    # peak_test_encoded = peaks_test
    # mfcc_test = mfcc_test
    

    print("################# Deuxième étape finitooooooo #################")

    print("Shape de chaque élément :")
    print(f"Shape X_train : {x_train.shape}")
    print(f"Shape X_test : {x_test.shape}")
    print(f"Shape X_train : {mfcc_train.shape}")
    print(f"Shape X_test : {mfcc_test.shape}")
    print(f"Shape y_train_encoded : {y_train_encoded.shape}")
    print(f"Shape y_test_encoded : {y_test_encoded.shape}")


    return x_train, y_train_encoded, peak_train, mfcc_train, x_test, y_test_encoded, peak_test_encoded, mfcc_test







# Packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score

# Nouveaux pacakges dont on a besoin pour : Test_train_hitnet_V2
import math
import random
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.activations import *
import datetime
import keras_tuner as kt
from tensorflow.keras import backend as K
from tensorflow.python.keras.saving import hdf5_format
import h5py
import json


from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback
import numpy as np

class F1Callback(Callback):
    """
    Callback Keras pour calculer le F1-score macro sur un jeu de validation à la fin
    de chaque epoch, le stocker et suivre la meilleure valeur obtenue.

    Args:
        x_val (np.ndarray): données d'entrée du jeu de validation.
        y_val (np.ndarray): étiquettes vraies du jeu de validation.

    Attributs:
        best_f1 (float): meilleur F1-score macro atteint jusqu'à présent.
        f1_per_epoch (list): historique du F1-score par epoch.
    """
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.best_f1 = 0
        self.f1_per_epoch = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.x_val), axis=1)
        score = f1_score(self.y_val, y_pred, average='macro')
        self.f1_per_epoch.append(score)
        logs['val_f1'] = score
        if score > self.best_f1:
            self.best_f1 = score





def performance_Hitnet_Que_Volant(model_name, x_train, y_train , x_val, y_val, output_path):

    """
    Entraîne et évalue un modèle CNN+GRU optimisé par recherche bayésienne
    pour la prédiction des coups (hit) dans des matchs de badminton.

    Étapes principales :
      - Définit un modèle CNN+GRU paramétrable pour Keras Tuner.
      - Recherche des hyperparamètres optimaux (BayesianOptimization).
      - Évalue les meilleurs modèles (F1-score macro, matrice de confusion, accuracy).
      - Sauvegarde les modèles, métriques détaillées et résultats dans un fichier Excel.
      - Calibre chaque modèle via temperature scaling et enregistre la température.

    Args:
        model_name (str): nom du modèle (pour affichage/sauvegarde).
        x_train (np.ndarray): caractéristiques d'entraînement.
        y_train (np.ndarray): labels d'entraînement.
        x_val (np.ndarray): caractéristiques de validation.
        y_val (np.ndarray): labels de validation.
        output_path (str): dossier de sortie pour les modèles et résultats.

    Returns:
        None
    """



    # On supprime la variable tuner si elle existe car nous cause des erreurs
    if 'tuner' in locals():
        del tuner
    
    num_consec = 12
    eps = 1e-8


    def build_cnn_gru_model(hp):
        input_layer = Input(shape=(x_train.shape[1],))
        x = Reshape((num_consec, x_train.shape[1] // num_consec))(input_layer)

        # --- Bloc CNN ---
        for i in range(hp.Int("num_conv_layers", 1, 2)):  # 1 ou 2 couches
            x = Conv1D(
                filters=hp.Choice(f"filters_{i}", [32, 64, 128]),
                kernel_size=hp.Choice(f"kernel_size_{i}", [3, 5]),
                activation="relu",
                padding="same"
            )(x)
            x = Dropout(hp.Choice(f"dropout_cnn_{i}", [0.0, 0.2, 0.4]))(x)

        # --- Bloc GRU ---
        for i in range(hp.Choice("gru_layers", [1, 2])):
            x = Bidirectional(GRU(
                units=hp.Choice("gru_units", [32, 64, 128]),
                return_sequences=(i < hp.get("gru_layers") - 1),  # Dernière couche : return_sequences=False
                dropout=hp.Choice("dropout_gru", [0.0, 0.2, 0.4]),
                kernel_regularizer=tf.keras.regularizers.l2(
                    hp.Choice("l2_reg", [0.0, 1e-6, 1e-4])
                )
            ))(x)

        # --- Fully connected ---
        x = Dense(
            units=hp.Choice("dense_units", [32, 64]),
            activation="relu"
        )(x)
        x = Dropout(hp.Choice("final_dropout", [0.0, 0.2, 0.4]))(x)

        # --- Output ---
        output = Dense(3, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=output)

        # --- Compile ---
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", [1e-2, 1e-3, 1e-4])
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model



    # Créez une instance de l'HyperModel avec le modèle de construction
    hypermodel = build_cnn_gru_model  #build_transformer_model # #build_cnn_model
    # Instanciez le tuner BayesianOptimization
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_accuracy',  # Métrique à optimiser
        max_trials=30, #30 ##TESTRAPIDE            # Nombre total d'essais
        directory='tuner_GRU_directory'  # Répertoire pour enregistrer les résultats
    )

    f1_callback = F1Callback(x_val, y_val)
    
    #changement de métrique d optimisation MATEO

    tuner.search(
        x_train, y_train,
        epochs=30, #50 ##TESTRAPIDE
        validation_data=(x_val, y_val),
        callbacks=[f1_callback]  
    )

    # Lancez la recherche d'hyperparamètres
    #tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

    best_models = tuner.get_best_models(num_models = 4)  #6 ##TESTRAPIDE

    modele_dict = {} # on définit le dictionnnaire
    print(f"on a {len(best_models)} modèles.") # on affiche le nombre de modele que l'on a
    for i in range(len(best_models)):
        modele_name = f'best_model_{i}' # on definit le nom du modele i 
        print(modele_name) # on l'affiche
        modele_dict[modele_name] = best_models[i] # on affecte le modele i à son nom dans le dic
        print(modele_dict[modele_name]) # on affiche la valeur du modele i 
        y_pred = np.argmax(modele_dict[modele_name].predict(x_val), axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        print(f"➡️ Best model {i} - val_f1: {f1:.4f}")
        
    


    confusmatrix_dict = {}
    metric_dict = {}
    for key, value in modele_dict.items():
        y_pred = np.argmax(value.predict(x_val), axis=1)
        print(f"########## Pour le modèle {key}, on a les résultats suivants : ##########")
        print("La valeur des différentes metrics :")
        print(classification_report(y_val, y_pred))
        metric_dict[key] = classification_report(y_val, y_pred) # on met les metrics dans le dic
        print("La matrice de confusion :")
        print(confusion_matrix(y_val, y_pred, normalize='true'))
        confusmatrix_dict[key] = confusion_matrix(y_val, y_pred, normalize='true') # on met la matrice dans le dic


    for key, value in metric_dict.items():  
        print(key)
        print(value)

    liste_de_dicts_GRU = []
    for key, value in metric_dict.items():
        liste_de_dicts_GRU.append(key)
    liste_de_dicts_GRU


    ############ Pour les f1 score
    # Création d'une liste qui contiendra des dictionnaires
    liste_de_dicts_GRU_f1 = []
    for key, value in metric_dict.items():
        
        print(key) #on affiche le nom du modèle
        print(value) # on affiche ses metrics
        # on divise la chaîne de caractères en lignes
        model_metrics_lines = value.split('\n')
        
        #### Pour la classe 0
        # Extraire le f1-score à partir de la 3ème ligne (index 2)
        f1_score_0 = model_metrics_lines[2].split()
        # L'indice correspondant au f1-score dépendra de la structure de la sortie,
        # mais généralement il s'agit du quatrième élément (index 3)
        f1_score_0 = float(f1_score_0[3])
        print("F1-score pour la classe 0 :", f1_score_0)

        #### Pour la classe 1
        # Extraire le f1-score à partir de la 4ème ligne (index 3)
        f1_score_1 = model_metrics_lines[3].split()
        # L'indice correspondant au f1-score dépendra de la structure de la sortie,
        # mais généralement il s'agit du quatrième élément (index 3)
        f1_score_1 = float(f1_score_1[3])
        print("F1-score pour la classe 1 :", f1_score_1)
        
        #### Pour la classe 2
        # Extraire le f1-score à partir de la 5ème ligne (index 4)
        f1_score_2 = model_metrics_lines[4].split()
        # L'indice correspondant au f1-score dépendra de la structure de la sortie,
        # mais généralement il s'agit du quatrième élément (index 3)
        f1_score_2 = float(f1_score_2[3])
        print("F1-score pour la classe 2 :", f1_score_2)
        
        # création du dictionnaire
        key = {f"F1-score de {key} pour la classe 0 :": f1_score_0, 
            f"F1-score de {key} pour la classe 1 :": f1_score_1,
            f"F1-score de {key} pour la classe 2 :": f1_score_2}
        
        # on ajoute le dictionnaire à la liste
        liste_de_dicts_GRU_f1.append(key)


    ############ Pour les matrice de confusion
    # Création d'une liste qui contiendra des dictionnaires
    liste_de_dicts_GRU_confusion= []
    for key, value in confusmatrix_dict.items():
        print(key) #on affiche le nom du modèle
        print(value) # on affiche ses metrics
        pourcent0 = value[0, 0] 
        pourcent1 = value[1, 1] 
        pourcent2 = value[2, 2] 
        print(pourcent0)
        print(pourcent1)
        # création du dictionnaire
        key = {f"pourcent0 de {key} :": pourcent0, 
            f"pourcent1 de {key} :": pourcent1,
            f"pourcent2 de {key} :": pourcent2}
        
        # on ajoute le dictionnaire à la liste
        liste_de_dicts_GRU_confusion.append(key)
        
    liste_de_dicts_GRU_confusion

    ############ Pour les Accuracy
    # Création d'une liste qui contiendra des dictionnaires
    liste_de_dicts_GRU_accuracy = []
    for key, value in metric_dict.items():
        
        print(key) #on affiche le nom du modèle
        print(value) # on affiche ses metrics
        # on divise la chaîne de caractères en lignes
        model_metrics_lines = value.split('\n')
        
        #### Pour l'accuracy
        # Extraire l'accuracy à partir de la 6ème ligne 
        accuracy = model_metrics_lines[6].split()
        accuracy = float(accuracy[1]) #2ème élément de la liste
        print("Accuracy du modèle :", accuracy)
        
        # création du dictionnaire
        key = {f"Accuracy du modèle :": accuracy}
        
        # on ajoute le dictionnaire à la liste
        liste_de_dicts_GRU_accuracy.append(key)

    liste_de_dicts_GRU_accuracy    



    #####################################################
    ################ Pour le modèle #####################
    #####################################################

    for idx, model in enumerate(best_models):
        model_filename = output_path +'/modele_que_volant_GRU'+ str(idx) + '.h5'
        model.save(model_filename)


    ####################################################
    ################ Pour le excel #####################
    ####################################################
    try:
        # Initialisez un DataFrame vide pour stocker les résultats
        results = pd.DataFrame(columns=['Model', 'F1-score (Class 0)', 'F1-score (Class 1)', 'F1-score (Class 2)', 'Class 0 %', 'Class 1 %', 'Class 2 %', 'Accuracy'])

        for name_modele, element_f1, element_confusion, element_accuracy in zip(liste_de_dicts_GRU, liste_de_dicts_GRU_f1, liste_de_dicts_GRU_confusion, liste_de_dicts_GRU_accuracy):
            # Créer un dictionnaire avec les résultats de l'itération actuelle
            result_dict = {
                'Model': name_modele,
                'F1-score (Class 0)': element_f1[f'F1-score de {name_modele} pour la classe 0 :'],
                'F1-score (Class 1)': element_f1[f'F1-score de {name_modele} pour la classe 1 :'],
                'F1-score (Class 2)': element_f1[f'F1-score de {name_modele} pour la classe 2 :'],
                'Class 0 %': round(element_confusion[f'pourcent0 de {name_modele} :'], 2),
                'Class 1 %': round(element_confusion[f'pourcent1 de {name_modele} :'], 2),
                'Class 2 %': round(element_confusion[f'pourcent2 de {name_modele} :'], 2),
                'Accuracy' : element_accuracy['Accuracy du modèle :']
            }

        

            # Ajouter le dictionnaire à DataFrame results en utilisant concat
            results = pd.concat([results, pd.DataFrame([result_dict])], ignore_index=True)

        # Afficher le DataFrame final
        results.head()

        excel_filename = output_path +'/Res_que_volant_GRU.xlsx'
        results.to_excel(excel_filename, index=False)
        print("c'est bien save")

    except Exception as error:
        print("Error while saving dataframe results:", error)
    
    ##########################################################
    ################ Pour la température #####################
    ##########################################################

    # Implémentation de la mise à l'échelle de la température pour l'étalonnage
    def temp_scaling(y_logits, y_val, max_iter=150):
        temp = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam()

        y_true = tf.convert_to_tensor(tf.keras.utils.to_categorical(y_val), dtype=tf.float32)

        print('Température - Valeur initiale :', temp.numpy())

        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                y_scaled = y_logits / temp
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=y_scaled, labels=y_true)
                )
            grads = tape.gradient(loss, [temp])
            optimizer.apply_gradients(zip(grads, [temp]))

        print('Température - Valeur finale :', temp.numpy())

        return temp.numpy()
    
    # Parcourir les clés et les valeurs du dictionnaire
    temp_dict = {} # on définit le dictionnnaire
    for key, value in modele_dict.items():
        print("Clé :", key, " - Valeur :", value)
        # compute_logits = K.function([value.layers[0].input], [value.layers[-2].output])  # Fonction pour calculer les logits
        # y_logits = compute_logits(x_val)[0]  # Calcul des logits pour l'ensemble de validation
        # Cherche une Dense avec units == 3 (nombre de classes) juste avant softmax
        for layer in reversed(value.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.units == 3:
                logits_layer = layer
                break
        logits_model = Model(inputs=value.input, outputs=logits_layer.output)

        y_logits = logits_model.predict(x_val)

        temp_dict[key] = temp_scaling(y_logits, y_val) # on met la température de la valeur finale dans le dic
        #temp = temp_scaling(y_logits, y_val)  # Étalonnage de la température
    print(temp_dict) # on affiche le dictionnaire


    
    for (key_model, value_model), (key_temp, value_temp) in zip(modele_dict.items(), temp_dict.items()):
        # 1. Sauver le modèle (format HDF5 ou Keras natif, à ta convenance)
        model_path = output_path + f'/temp_hitnet_que_volant_GRU{key_model}-{num_consec}.keras'
        value_model.save(model_path)

        # 2. Sauver la température dans un fichier séparé (JSON propre)
        temp_path = output_path + f'/temp_hitnet_que_volant_GRU{key_model}-{num_consec}_temp.json'
        with open(temp_path, 'w') as temp_file:
            json.dump({'temperature': float(value_temp)}, temp_file)




class F1Callback2(Callback):
    """
    Callback Keras pour calculer le F1-score macro sur deux entrées
    (trajectoire + MFCC) à la fin de chaque epoch, suivre la meilleure
    valeur atteinte et arrêter l'entraînement si le score reste faible.

    Args:
        x_val_traj (np.ndarray): données de validation trajectoires.
        x_val_mfcc (np.ndarray): données de validation MFCC.
        y_val (np.ndarray): étiquettes vraies du jeu de validation.
    """
    def __init__(self, x_val_traj, x_val_mfcc, y_val):
        super().__init__()
        self.x_val_traj = x_val_traj
        self.x_val_mfcc = x_val_mfcc
        self.y_val = y_val
        self.best_f1 = 0
        self.f1_per_epoch = []
        self.no_improvement_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict([self.x_val_traj, self.x_val_mfcc]), axis=1)
        score = f1_score(self.y_val, y_pred, average='macro')
        self.f1_per_epoch.append(score)
        logs['val_f1'] = score
        if score > self.best_f1:
            self.best_f1 = score
        # Skip if stuck too low
        if epoch >= 5 and max(self.f1_per_epoch) < 0.25:
            print("⚠️ Trial stopped early (low F1 score)")
            self.model.stop_training = True
        
        

        







def performance_Hitnet_Volant_Son(model_name, x_train, y_train, x_val, y_val, mfcc_train, mfcc_val, output_path):
    """
    Entraîne et évalue un modèle bimodal CNN+GRU (entrée trajectoire + entrée audio MFCC ou spectrogramme)
    optimisé par recherche bayésienne pour la prédiction des coups (hit) au badminton.

    Étapes principales :
      - Définit deux architectures CNN+GRU (une pour MFCC 1D, une pour spectrogramme 2D) utilisables par Keras Tuner.
      - Lance une recherche bayésienne d’hyperparamètres (BayesianOptimization) avec callback F1.
      - Évalue les meilleurs modèles (F1-score, matrice de confusion, accuracy par classe).
      - Sauvegarde les modèles, métriques et un fichier Excel récapitulatif.
      - Calibre chaque modèle par temperature scaling et enregistre la température.

    Args:
        model_name (str): nom du modèle (pour affichage/sauvegarde).
        x_train (np.ndarray): caractéristiques trajectoire d’entraînement.
        y_train (np.ndarray): labels d’entraînement.
        x_val (np.ndarray): caractéristiques trajectoire de validation.
        y_val (np.ndarray): labels de validation.
        mfcc_train (np.ndarray): caractéristiques audio d’entraînement (MFCC ou spectrogrammes).
        mfcc_val (np.ndarray): caractéristiques audio de validation.
        output_path (str): dossier de sortie pour les modèles et résultats.

    Returns:
        None
    """
    # On supprime la variable tuner si elle existe car nous cause des erreurs
    if 'tuner' in locals():
        del tuner
    
    num_consec = 12
    eps = 1e-8
    print(f"MMDMDMMDDD {mfcc_train.shape}, {x_train.shape}")
    # mfcc_train_temporel = creer_mfcc_temporel_centre(mfcc_train, num_consec=12)
    # mfcc_val_temporel = creer_mfcc_temporel_centre(mfcc_val, num_consec=12)


    def build_2_cnn_gru_model(hp):
        # --- Entrée trajectoire ---
        traj_input = Input(shape=(x_train.shape[1],), name="traj_input")
        traj_reshaped = Reshape((num_consec, x_train.shape[1] // num_consec))(traj_input)

        x_traj = traj_reshaped
        for i in range(hp.Int("num_conv_layers", 1, 2,3)):
            x_traj = Conv1D(
                filters=hp.Choice(f"filters_traj_{i}", [32, 64,96, 128]),
                kernel_size=hp.Choice(f"kernel_size_traj_{i}", [3,4,5,6]),
                activation="relu",
                padding="same"
            )(x_traj)
            x_traj = Dropout(hp.Choice(f"dropout_traj_{i}", [0.0, 0.05, 0.1, 0.15, 0.2]))(x_traj)

        # --- Entrée MFCC ---
        # Forme attendue : (batch_size, num_consec, num_mfcc)
        # mfcc_input = Input(shape=(num_consec, mfcc_train_temporel.shape[2]), name="mfcc_input")
        mfcc_input = Input(shape=(mfcc_train.shape[1], mfcc_train.shape[2]), name="mfcc_input")
        x_mfcc = mfcc_input
        for i in range(hp.Int("num_conv_layers_mfcc", 1, 2)):
            x_mfcc = Conv1D(
                filters=hp.Choice(f"filters_mfcc_{i}", [32, 64, 96, 128]),
                kernel_size=hp.Choice(f"kernel_size_mfcc_{i}", [3, 4, 5, 6]),
                activation="relu",
                padding="same"
            )(x_mfcc)
            x_mfcc = Dropout(hp.Choice(f"dropout_mfcc_{i}", [0.0, 0.05, 0.1, 0.15, 0.2]))(x_mfcc)

        # --- Fusion CNNs ---
        x = Concatenate(axis=-1)([x_traj, x_mfcc])

        # --- Bloc GRU ---
        for i in range(hp.Choice("gru_layers", [1, 2])):
            x = Bidirectional(GRU(
                units=hp.Choice("gru_units", [32, 64, 96, 128, 160]),
                return_sequences=(i < hp.get("gru_layers") - 1),
                dropout=hp.Choice("dropout_gru", [0.0, 0.1, 0.2, 0.3, 0.4]),
                kernel_regularizer=tf.keras.regularizers.l2(
                    hp.Choice("l2_reg", [0.0, 1e-6,1e-5, 1e-4])
                )
            ))(x)

        # --- Dense ---
        x = Dense(
            units=hp.Choice("dense_units", [48, 64, 80, 96]),
            activation="relu"
        )(x)
        x = Dropout(hp.Choice("final_dropout", [0.0, 0.05, 0.1, 0.15, 0.2]))(x)

        # --- Sortie ---
        output = Dense(3, activation="softmax")(x)

        # --- Compilation ---
        model = Model(inputs=[traj_input, mfcc_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", [1e-3, 5e-3, 1e-4])
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    

    def build_2_cnn_gru_model_2(hp):
        # --- Entrée trajectoire ---
        traj_input = Input(shape=(x_train.shape[1],), name="traj_input")
        traj_reshaped = Reshape((num_consec, x_train.shape[1] // num_consec))(traj_input)

        x_traj = traj_reshaped
        for i in range(hp.Int("num_conv_layers", 1, 2,3)):
            x_traj = Conv1D(
                filters=hp.Choice(f"filters_traj_{i}", [32, 64,96, 128]),
                kernel_size=hp.Choice(f"kernel_size_traj_{i}", [3,4,5,6]),
                activation="relu",
                padding="same"
            )(x_traj)
            x_traj = Dropout(hp.Choice(f"dropout_traj_{i}", [0.0, 0.05, 0.1, 0.15, 0.2]))(x_traj)


        # Étape 1 : choisir un string
        kernel_size_str = hp.Choice(f"kernel_size_spec_{i}", ["3x3", "3x5", "5x3", "5x5"])

        # Étape 2 : convertir en tuple
        kernel_map = {
            "3x3": (3, 3),
            "3x5": (3, 5),
            "5x3": (5, 3),
            "5x5": (5, 5)
        }
        kernel_size = kernel_map[kernel_size_str]
        # --- Entrée MFCC ---
        # Forme attendue : (batch_size, num_consec, num_mfcc)
        # mfcc_input = Input(shape=(num_consec, mfcc_train_temporel.shape[2]), name="mfcc_input")
        mfcc_input = Input(shape=(12, 128, 6), name="spec_input")  # (T, Freq, TimePerFrame)

        x_spec = mfcc_input
        
        x_spec = Reshape((12, 128, 6, 1))(x_spec)
        for i in range(hp.Int("num_conv_layers_spec", 1, 3)):
            x_spec = TimeDistributed(Conv2D(
                filters=hp.Choice(f"filters_spec_{i}", [32, 64, 96, 128]),
                kernel_size=kernel_size,
                activation="relu",
                padding="same"
            ))(x_spec)
            x_spec = Dropout(hp.Choice(f"dropout_spec_{i}", [0.0, 0.1, 0.2]))(x_spec)
        x_spec = TimeDistributed(GlobalAveragePooling2D())(x_spec)

        # --- Fusion CNNs --- # Devient (None, 12 * nb_filtres)
        x = Concatenate(axis=-1)([x_traj, x_spec])

        # --- Bloc GRU ---
        for i in range(hp.Choice("gru_layers", [1, 2])):
            x = Bidirectional(GRU(
                units=hp.Choice("gru_units", [32, 64, 96, 128, 160]),
                return_sequences=(i < hp.get("gru_layers") - 1),
                dropout=hp.Choice("dropout_gru", [0.0, 0.1, 0.2, 0.3, 0.4]),
                kernel_regularizer=tf.keras.regularizers.l2(
                    hp.Choice("l2_reg", [0.0, 1e-6,1e-5, 1e-4])
                )
            ))(x)

        # --- Dense ---
        x = Dense(
            units=hp.Choice("dense_units", [48, 64, 80, 96]),
            activation="relu"
        )(x)
        x = Dropout(hp.Choice("final_dropout", [0.0, 0.05, 0.1, 0.15, 0.2]))(x)

        # --- Sortie ---
        output = Dense(3, activation="softmax")(x)

        # --- Compilation ---
        model = Model(inputs=[traj_input, mfcc_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", [1e-3, 5e-3, 1e-4])
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model


    # Créez une instance de l'HyperModel avec le modèle de construction
    hypermodel = build_2_cnn_gru_model_2 #build_2_cnn_gru_model  #build_transformer_model # #build_cnn_model
    # Instanciez le tuner BayesianOptimization
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_accuracy',  # Métrique à optimiser
        max_trials=30, #30 ##TESTRAPIDE            # Nombre total d'essais
        directory='tuner_2_CNN_directory_2'  # Répertoire pour enregistrer les résultats
    )

    f1_callback = F1Callback2(x_val, mfcc_val, y_val)
    
    #changement de métrique d optimisation MATEO

    tuner.search(
        [x_train, mfcc_train], y_train,
        epochs=30, #50 ##TESTRAPIDE
        validation_data=([x_val, mfcc_val], y_val),
        callbacks=[f1_callback]  
    )

    # Lancez la recherche d'hyperparamètres
    #tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

    best_models = tuner.get_best_models(num_models = 4)  #6 ##TESTRAPIDE

    modele_dict = {} # on définit le dictionnnaire
    print(f"on a {len(best_models)} modèles.") # on affiche le nombre de modele que l'on a
    for i in range(len(best_models)):
        modele_name = f'best_model_{i}' # on definit le nom du modele i 
        print(modele_name) # on l'affiche
        modele_dict[modele_name] = best_models[i] # on affecte le modele i à son nom dans le dic
        print(modele_dict[modele_name]) # on affiche la valeur du modele i 
        y_pred = np.argmax(modele_dict[modele_name].predict([x_val, mfcc_val]), axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        print(f"➡️ Best model {i} - val_f1: {f1:.4f}")
        
    


    confusmatrix_dict = {}
    metric_dict = {}
    for key, value in modele_dict.items():
        y_pred = np.argmax(value.predict([x_val, mfcc_val]), axis=1)
        print(f"########## Pour le modèle {key}, on a les résultats suivants : ##########")
        print("La valeur des différentes metrics :")
        print(classification_report(y_val, y_pred))
        metric_dict[key] = classification_report(y_val, y_pred) # on met les metrics dans le dic
        print("La matrice de confusion :")
        print(confusion_matrix(y_val, y_pred, normalize='true'))
        confusmatrix_dict[key] = confusion_matrix(y_val, y_pred, normalize='true') # on met la matrice dans le dic


    for key, value in metric_dict.items():  
        print(key)
        print(value)

    liste_de_dicts_GRU = []
    for key, value in metric_dict.items():
        liste_de_dicts_GRU.append(key)
    liste_de_dicts_GRU


    ############ Pour les f1 score
    # Création d'une liste qui contiendra des dictionnaires
    liste_de_dicts_GRU_f1 = []
    for key, value in metric_dict.items():
        
        print(key) #on affiche le nom du modèle
        print(value) # on affiche ses metrics
        # on divise la chaîne de caractères en lignes
        model_metrics_lines = value.split('\n')
        
        #### Pour la classe 0
        # Extraire le f1-score à partir de la 3ème ligne (index 2)
        f1_score_0 = model_metrics_lines[2].split()
        # L'indice correspondant au f1-score dépendra de la structure de la sortie,
        # mais généralement il s'agit du quatrième élément (index 3)
        f1_score_0 = float(f1_score_0[3])
        print("F1-score pour la classe 0 :", f1_score_0)

        #### Pour la classe 1
        # Extraire le f1-score à partir de la 4ème ligne (index 3)
        f1_score_1 = model_metrics_lines[3].split()
        # L'indice correspondant au f1-score dépendra de la structure de la sortie,
        # mais généralement il s'agit du quatrième élément (index 3)
        f1_score_1 = float(f1_score_1[3])
        print("F1-score pour la classe 1 :", f1_score_1)
        
        #### Pour la classe 2
        # Extraire le f1-score à partir de la 5ème ligne (index 4)
        f1_score_2 = model_metrics_lines[4].split()
        # L'indice correspondant au f1-score dépendra de la structure de la sortie,
        # mais généralement il s'agit du quatrième élément (index 3)
        f1_score_2 = float(f1_score_2[3])
        print("F1-score pour la classe 2 :", f1_score_2)
        
        # création du dictionnaire
        key = {f"F1-score de {key} pour la classe 0 :": f1_score_0, 
            f"F1-score de {key} pour la classe 1 :": f1_score_1,
            f"F1-score de {key} pour la classe 2 :": f1_score_2}
        
        # on ajoute le dictionnaire à la liste
        liste_de_dicts_GRU_f1.append(key)


    ############ Pour les matrice de confusion
    # Création d'une liste qui contiendra des dictionnaires
    liste_de_dicts_GRU_confusion= []
    for key, value in confusmatrix_dict.items():
        print(key) #on affiche le nom du modèle
        print(value) # on affiche ses metrics
        pourcent0 = value[0, 0] 
        pourcent1 = value[1, 1] 
        pourcent2 = value[2, 2] 
        print(pourcent0)
        print(pourcent1)
        # création du dictionnaire
        key = {f"pourcent0 de {key} :": pourcent0, 
            f"pourcent1 de {key} :": pourcent1,
            f"pourcent2 de {key} :": pourcent2}
        
        # on ajoute le dictionnaire à la liste
        liste_de_dicts_GRU_confusion.append(key)
        
    liste_de_dicts_GRU_confusion

    ############ Pour les Accuracy
    # Création d'une liste qui contiendra des dictionnaires
    liste_de_dicts_GRU_accuracy = []
    for key, value in metric_dict.items():
        
        print(key) #on affiche le nom du modèle
        print(value) # on affiche ses metrics
        # on divise la chaîne de caractères en lignes
        model_metrics_lines = value.split('\n')
        
        #### Pour l'accuracy
        # Extraire l'accuracy à partir de la 6ème ligne 
        accuracy = model_metrics_lines[6].split()
        accuracy = float(accuracy[1]) #2ème élément de la liste
        print("Accuracy du modèle :", accuracy)
        
        # création du dictionnaire
        key = {f"Accuracy du modèle :": accuracy}
        
        # on ajoute le dictionnaire à la liste
        liste_de_dicts_GRU_accuracy.append(key)

    liste_de_dicts_GRU_accuracy    



    #####################################################
    ################ Pour le modèle #####################
    #####################################################

    for idx, model in enumerate(best_models):
        model_filename = output_path +'/modele_que_volant_GRU'+ str(idx) + '.h5'
        model.save(model_filename)


    ####################################################
    ################ Pour le excel #####################
    ####################################################
    try:
        # Initialisez un DataFrame vide pour stocker les résultats
        results = pd.DataFrame(columns=['Model', 'F1-score (Class 0)', 'F1-score (Class 1)', 'F1-score (Class 2)', 'Class 0 %', 'Class 1 %', 'Class 2 %', 'Accuracy'])

        for name_modele, element_f1, element_confusion, element_accuracy in zip(liste_de_dicts_GRU, liste_de_dicts_GRU_f1, liste_de_dicts_GRU_confusion, liste_de_dicts_GRU_accuracy):
            # Créer un dictionnaire avec les résultats de l'itération actuelle
            result_dict = {
                'Model': name_modele,
                'F1-score (Class 0)': element_f1[f'F1-score de {name_modele} pour la classe 0 :'],
                'F1-score (Class 1)': element_f1[f'F1-score de {name_modele} pour la classe 1 :'],
                'F1-score (Class 2)': element_f1[f'F1-score de {name_modele} pour la classe 2 :'],
                'Class 0 %': round(element_confusion[f'pourcent0 de {name_modele} :'], 2),
                'Class 1 %': round(element_confusion[f'pourcent1 de {name_modele} :'], 2),
                'Class 2 %': round(element_confusion[f'pourcent2 de {name_modele} :'], 2),
                'Accuracy' : element_accuracy['Accuracy du modèle :']
            }

        

            # Ajouter le dictionnaire à DataFrame results en utilisant concat
            results = pd.concat([results, pd.DataFrame([result_dict])], ignore_index=True)

        # Afficher le DataFrame final
        results.head()

        excel_filename = output_path +'/Res_que_volant_GRU.xlsx'
        results.to_excel(excel_filename, index=False)
        print("c'est bien save")

    except Exception as error:
        print("Error while saving dataframe results:", error)
    
    ##########################################################
    ################ Pour la température #####################
    ##########################################################

    # Implémentation de la mise à l'échelle de la température pour l'étalonnage
    def temp_scaling(y_logits, y_val, max_iter=150):
        temp = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam()

        y_true = tf.convert_to_tensor(tf.keras.utils.to_categorical(y_val), dtype=tf.float32)

        print('Température - Valeur initiale :', temp.numpy())

        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                y_scaled = y_logits / temp
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=y_scaled, labels=y_true)
                )
            grads = tape.gradient(loss, [temp])
            optimizer.apply_gradients(zip(grads, [temp]))

        print('Température - Valeur finale :', temp.numpy())

        return temp.numpy()
    
    # Parcourir les clés et les valeurs du dictionnaire
    temp_dict = {} # on définit le dictionnnaire
    for key, value in modele_dict.items():
        print("Clé :", key, " - Valeur :", value)
        # compute_logits = K.function([value.layers[0].input], [value.layers[-2].output])  # Fonction pour calculer les logits
        # y_logits = compute_logits(x_val)[0]  # Calcul des logits pour l'ensemble de validation
        # Cherche une Dense avec units == 3 (nombre de classes) juste avant softmax
        for layer in reversed(value.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.units == 3:
                logits_layer = layer
                break
        logits_model = Model(inputs=value.input, outputs=logits_layer.output)

        y_logits = logits_model.predict([x_val, mfcc_val])

        temp_dict[key] = temp_scaling(y_logits, y_val) # on met la température de la valeur finale dans le dic
        #temp = temp_scaling(y_logits, y_val)  # Étalonnage de la température
    print(temp_dict) # on affiche le dictionnaire


    
    for (key_model, value_model), (key_temp, value_temp) in zip(modele_dict.items(), temp_dict.items()):
        # 1. Sauver le modèle (format HDF5 ou Keras natif, à ta convenance)
        model_path = output_path + f'/temp_hitnet_que_volant_GRU{key_model}-{num_consec}.keras'
        value_model.save(model_path)

        # 2. Sauver la température dans un fichier séparé (JSON propre)
        temp_path = output_path + f'/temp_hitnet_que_volant_GRU{key_model}-{num_consec}_temp.json'
        with open(temp_path, 'w') as temp_file:
            json.dump({'temperature': float(value_temp)}, temp_file)