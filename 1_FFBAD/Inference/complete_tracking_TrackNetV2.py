import os
import sys
from tqdm import tqdm
from TrackNetv2.three_in_three_out.predict3 import *
import pandas as pd
import pickle
import argparse
import cv2
import numpy as np
import time
from camera_changes import *
from extract_trajectoire import *
import glob
import torch
from torch.utils.data import Dataset, DataLoader

import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.path.append('/home/onyxia/work/mon-projet/sons')
from torch.utils.data import DataLoader
from extract_sons_pipeline import *
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable


# Fonction pour extraire une frame de référence à partir de la vidéoqui va etre utile pour reconnaitre
# les scènes a analyser. Actuellement on utilise la deuxième Frame de la vidéo comme référence mais à changer
# par la frame demander par l'interface
def extract_reference_frame(video_path, output_path='reference.jpg', time_sec=2):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la vidéo.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_path, frame)
        print(f"✅ Image de référence générée : {output_path}")
    else:
        print("❌ Impossible de lire la frame.")
    cap.release()


# Classe permettant d'associer les features de son associé au pic extrait avant d entrer dans le model.
class MelSpecWithFeatDataset(Dataset):
    def __init__(self,
                 spec_dir,
                 feat_dir,
                 split=None,
                 pca_n=30,
                 seed=42):

        # Chemins directs (tu peux aussi les passer directement comme args si besoin)
        spec_path = os.path.join(spec_dir, "mel_specs.npy")
        feat_path = os.path.join(feat_dir, "feats.csv")  

        # Charger les spectrogrammes
        specs = np.load(spec_path, allow_pickle=True)
        if specs.dtype == object:
            specs = np.stack(specs, axis=0)

        # Charger les features
        df_feat = pd.read_csv(feat_path)
        drop_cols = ['Frame', 'Time (s)', 'PEAK_HIT']
        feat_cols = [c for c in df_feat.columns if c not in drop_cols]
        df_feat[feat_cols] = df_feat[feat_cols].fillna(df_feat[feat_cols].mean())
        feats = df_feat[feat_cols].to_numpy(dtype=np.float32)

        # Normalisation
        scaler = StandardScaler()
        feats = scaler.fit_transform(feats)

        # Vérification dimensions
        if specs.shape[0] != feats.shape[0]:
            raise ValueError(f"Dimensions incompatibles : {specs.shape[0]} specs vs {feats.shape[0]} feats")

        # PCA optionnel
        if pca_n is not None and pca_n > 0:
            pca = PCA(n_components=pca_n, random_state=seed)
            feats = pca.fit_transform(feats)

        # Conversion
        self.mels = torch.from_numpy(specs.astype(np.float32)).unsqueeze(1)
        self.feats = torch.from_numpy(feats.astype(np.float32))

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, i):
        return self.mels[i], self.feats[i]


# Fonction permettant d'ajuster la prédiction issu des trajectoire à celle issu du son
def boosting_with_aligned(y_pred_proba: np.ndarray,
                          aligned: np.ndarray,
                          threshold: float,
                          boost_factor: float) -> np.ndarray:
    """
    Booste les probabilités des classes 1 et 2 là où aligned > threshold.

    Parameters
    ----------
    y_pred_proba : np.ndarray, shape (N,3)
        Probabilités initiales pour [no, bottom, top].
    aligned : np.ndarray, shape (N,)
        Signal de peak (0 à 1) indiquant la probabilité d'un hit.
    threshold : float
        Seuil au‐delà duquel on considère qu'il y a un hit à booster.
    boost_factor : float
        Facteur de boost appliqué aux probabilités bottom/top.

    Returns
    -------
    y_boosted : np.ndarray, shape (N,3)
        Probabilités après boost et re‐normalisation.
    """
    y_boosted = y_pred_proba.copy()
    N = y_boosted.shape[0]
    assert aligned.shape[0] == N, "aligned doit avoir même longueur que y_pred_proba"

    # On booste bottom et top là où aligned dépasse le seuil
    mask = aligned > threshold
    y_boosted[mask, 1] *= boost_factor
    y_boosted[mask, 2] *= boost_factor

    # Re‐normalisation ligne par ligne
    sums = y_boosted.sum(axis=1, keepdims=True)
    # Pour éviter division par zéro
    sums[sums == 0] = 1.0
    y_boosted /= sums

    return y_boosted


#La plus par des vidéos présentes un décalage son image cette fonction permet d aligner les prédiction son et trajectoire.
#Le shift optimal est celui qui maximisera l alignement des 2 sources de prédictions.
def find_optimal_shift(pred_proba, target, threshold=0.6, max_shift=10):
    """
    Trouve le décalage optimal entre les prédictions (multiclasses) et le target aligné (binaire on associe top et bottom pour les traj) .
    """
    best_shift = 0
    best_score = -1

    # Réduire la prédiction multiclasse à une seule valeur de proba positive (bottom + top)
    # On prend la max entre classe 1 (bottom) et classe 2 (top)
    positive_proba = np.maximum(pred_proba[:, 1], pred_proba[:, 2])
    pred_bin = (positive_proba > threshold).astype(int)

    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            shifted_target = np.pad(target[-shift:], (0, -shift), mode='constant')
        elif shift > 0:
            shifted_target = np.pad(target[:-shift], (shift, 0), mode='constant')
        else:
            shifted_target = target.copy()

        target_bin = (shifted_target > threshold).astype(int)
        score = np.sum(pred_bin & target_bin)

        if score > best_score:
            best_score = score
            best_shift = shift

    return best_shift


def shift_array(arr, shift):
    if shift < 0:
        return np.pad(arr[-shift:], (0, -shift), mode='constant')
    elif shift > 0:
        return np.pad(arr[:-shift], (shift, 0), mode='constant')
    else:
        return arr.copy()




WIDTH = 512
HEIGHT = 288
BATCH_SIZE = 1


def lister_fichiers(repertoire, extensions):
    fichiers = []
    for dossier, sous_dossiers, fichiers_dans_dossier in os.walk(repertoire):
        for fichier in fichiers_dans_dossier:
            nom, extension = os.path.splitext(fichier)
            if extension.lower() in extensions:
                fichiers.append(nom + extension)
    fichiers.sort()
    return fichiers



#Fonction qui effectue Tracking du volant, extraction des features de son, préprocessing des données, inférence, puis prédiction finale
def complete_tracking(inputs_path, videos, references, outputs_path):
    #Boucle sur les différentes vidéos (A tester)
    for videoName, reference in zip(videos, references):

        video_path = os.path.join(inputs_path, videoName)
        reference_frame_path = os.path.join(inputs_path, reference)
        print("Reference path:", reference_frame_path)
        dirpath = os.path.dirname(os.path.abspath(__file__))
        print(os.path.exists(dirpath + "/TrackNetv2/three_in_three_out/model906_30.h5"))
        # Charger le modèle TrackNetV2
        model_tracknet = load_model(
            dirpath + "/TrackNetv2/three_in_three_out/model906_30.h5",
            custom_objects={'custom_loss': custom_loss},
            compile=False
            )


        # Détection des scènes avec angle classique
        print("Beginning detecting similar scenes")
        similar_scenes = detect_similar_camera(video_path, reference_frame_path)
        print(f"{len(similar_scenes)} scènes détectées")
        
        print("Done...")

        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        ret, image1 = cap.read()
        ratio = image1.shape[0] / HEIGHT
        size = (int(WIDTH * ratio), int(HEIGHT * ratio))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(outputs_path, videoName[:-4] + '_predict.mp4'), fourcc, fps, size)

        print('Beginning predicting......')
        start = time.time()

        # DataFrame pour stocker les résultats TrackNet
        df_predict_tracknet = pd.DataFrame(columns=['Frame', 'Visibility_Shuttle', 'ShuttleX', 'ShuttleY', 'Time'])
        df_predict_tracknet = df_predict_tracknet.set_index('Frame')

        from tqdm import tqdm

        total_scenes = len(similar_scenes)
        print(f"Nombre total de scènes à traiter : {total_scenes}")

        # Barre de progression pour les scènes
        for idx_scene, scene in enumerate(tqdm(similar_scenes, desc="Scènes traitées", unit="scène")):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()

            # Afficher la scène en cours
            print(f"\n→ Scène {idx_scene + 1}/{total_scenes} : frames {start_frame} à {end_frame}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Barre de progression pour les frames de la scène
            frame_range = range(start_frame, end_frame + 1, 3)
            for frame_number in tqdm(frame_range, desc=f"  Frames scène {idx_scene + 1}", leave=False, unit="frame"):
                success, image1 = cap.read()
                frame_time1 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
                success, image2 = cap.read()
                frame_time2 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
                success, image3 = cap.read()
                frame_time3 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))

                if not success:
                    break

                ### Tracking du volant ###
                unit = []
                # Prépare une séquence de 3 images avec leurs canaux séparés, comme entrée pour le modèle de prédiction.
                for img in [image1, image2, image3]:
                    x = img[..., ::-1]
                    x = array_to_img(x)
                    x = x.resize((WIDTH, HEIGHT))
                    x = np.moveaxis(img_to_array(x), -1, 0)
                    unit.extend([x[0], x[1], x[2]])
                # Transformation pour le model
                unit = np.asarray(unit).reshape((1, 9, HEIGHT, WIDTH)).astype('float32') / 255
                #Prédiction du masque de position du volant sur les 3 images
                y_pred = model_tracknet.predict(unit, batch_size=BATCH_SIZE, verbose=0)
                y_pred = (y_pred > 0.5).astype('float32')
                h_pred = (y_pred[0] * 255).astype('uint8') # Convertit en image 8-bit pour traitement OpenCV

                for i, (image, frame_time) in enumerate(zip([image1, image2, image3],
                                                        [frame_time1, frame_time2, frame_time3])):
                    # Si aucune prédiction détectée (aucun pixel actif), on enregistre 0 (pas de volant).
                    if np.amax(h_pred[i]) <= 0:
                        df_predict_tracknet.loc[frame_number + i] = [0, 0, 0, frame_time]
                    # Si une zone est détectée : on l’analyse, calcule la position du volant, l’enregistre et la dessine.
                    else:
                        cnts, _ = cv2.findContours(h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        rects = [cv2.boundingRect(ctr) for ctr in cnts]
                        target = max(rects, key=lambda r: r[2] * r[3])
                        cx_pred = int(ratio * (target[0] + target[2] / 2))
                        cy_pred = int(ratio * (target[1] + target[3] / 2))
                        df_predict_tracknet.loc[frame_number + i] = [1, cx_pred, cy_pred, frame_time]
                        cv2.circle(image, (cx_pred, cy_pred), 5, (0, 0, 255), -1) # Dessine un cercle sur l’image pour visualiser le suivi.
                    out.write(image) # Écrit chaque frame dans la vidéo finale.
        out.release()
        end = time.time()
        print('Prediction time:', end - start, 'secs')
        print('Done......')

        cap.release()
            
        # nettoye le DataFrame après traitement, pour ne garder qu’une seule entrée par frame
        df_predict_tracknet = df_predict_tracknet[~df_predict_tracknet.index.duplicated(keep='first')]
        print("Détection de trajectoire Done")


        #Extraction des pics sonores et renvoie un dataset avec les Frame associé aux pics
        audio_path = os.path.join(outputs_path, videoName[:-4] + ".wav")
        df_dataset, df_peaks, fps, sr, y = peak_extractor(
                chemin_audio=audio_path,
                chemin_video=video_path,
                chemin_df=df_predict_tracknet
            )
        # Extrait des features de sons associé à chaques pics
        features_extractor(
            df_peaks=df_peaks,
            fps=fps, sr=sr, y=y,
            chemin_sauvegarde_csv=os.path.join(outputs_path, videoName[:-4] + "_feat.csv")
        )

        #Extrait les images de log-mel spectrogrammes associés à chaque pics
        spectre_extractor(
                y=y, sr=sr,
                df_peaks=df_peaks,
                save_dir=outputs_path
            )

        # Préparation du dataframe avant de rentrer dans le model
        dataset = MelSpecWithFeatDataset(
            spec_dir=outputs_path,
            feat_dir=outputs_path,
            split=None,
            pca_n=30
        )
        # Mettre dans un DataLoader pour le batch
        loader = DataLoader(dataset, batch_size=64)
        # Charger le modèle complet
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_path = os.path.join(os.path.dirname(__file__), "strike_cnn_feat_new.pt")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        # Prédiction
        all_probs = []
        with torch.no_grad():
            for x_seq, feat_seq in loader:
                x_seq, feat_seq = x_seq.to(device), feat_seq.to(device)
                logits = model(x_seq, feat_seq) 
                probs = torch.softmax(logits, dim=1)[:, 1]  # probabilité classe 1
                all_probs.extend(probs.cpu().numpy())

        peak_hit_prob = np.zeros(len(df_dataset))
        # Obtenir les indices où Peak_binary == 1
        peak_indices = df_dataset.index[df_dataset["Peak_binary"] == 1].tolist()
        # Assigner les valeurs de all_probs aux bons indices
        peak_hit_prob[peak_indices] = all_probs  # Assumes all_probs is aligned with those indices
        # Ajouter la colonne au DataFrame
        df_dataset["peak_hit_prob"] = peak_hit_prob




        @register_keras_serializable()
        class GRUWithTimeMajor(tf.keras.layers.GRU):
            def __init__(self, *args, time_major=False, **kwargs):
                super().__init__(*args, **kwargs)

        
        model_path_GRU =  os.path.join(os.path.dirname(__file__), "modele_que_volant_CNN0.h5")
        # Préprocessing des coordonnées du volant avant inférence 
        df_traj = pre_processing_for_inference(df_dataset)
        traj_path = os.path.join(outputs_path, "traj.npy")
        np.save(traj_path, df_traj)
        print("✅ df_traj sauvegardé sous forme de fichier .npy.")

        model_files_GRU_CNN = glob.glob(os.path.join(model_path_GRU, 'modele_que_volant_CNN0.h5'))
        model_file_GRU_CNN = model_files_GRU_CNN[0]

        model_GRU_CNN = tf.keras.models.load_model(
                    model_file_GRU_CNN,
                    custom_objects={'GRU': GRUWithTimeMajor}
                )

        #Prédiction issu des trajectoires
        y_pred_proba_GRU_CNN = model_GRU_CNN.predict(df_traj)
        
        #On retire les 11 premières valeurs de peak_hit_prob pour l'aligner temporellement avec les prédictions 
        aligned = peak_hit_prob[11:] 

        #Différents paramètres modulables en fonction de la confiance des prédictions, de la qualité du son et du recall souhaité.

        biais= 0.7        # Poids donné à la probabilité issue des pics détectés
        boost= 1          # Intensité du boosting appliqué aux pics
        max_dist= 4       # Fenêtre de voisinage pour étendre les prédictions d'un pic local
        ratio= 0.4        # Poids de fusion entre prédiction GRU-CNN et pic aligné
        tresh= 0.35       # Seuil minimal de probabilité pour considérer un pic

        # Pour chaque point, on prend le maximum dans une fenêtre glissante centrée sur ce point.
        aligned_extend = np.zeros_like(aligned)
        for i in range(len(aligned)):
            start = max(0, i - max_dist)
            end   = min(len(aligned), i + max_dist + 1)
            aligned_extend[i] = np.max(aligned[start:end])

        # Construction d’une probabilité "prior" à partir des pics pour avoir les dimension que les prédictions issues de trajectoires
        prob_peak = np.zeros_like(y_pred_proba_GRU_CNN)
        prob_peak[:,1] = aligned_extend * biais
        prob_peak[:,2] = aligned_extend * biais
        prob_peak[:,0] = 1 - prob_peak[:,1] - prob_peak[:,2]

        # Décalage temporel optimal
        optimal_shift = find_optimal_shift(y_pred_proba_GRU_CNN, aligned_extend, threshold=0.6, max_shift=10)
        print("Décalage optimal :", optimal_shift)
        aligned_shifted = shift_array(aligned_extend, optimal_shift)

        # Boosting avec la version décalée
        y_boosted = boosting_with_aligned(y_pred_proba_GRU_CNN, aligned_shifted, tresh, boost)

        fused = ratio * prob_peak + (1 - ratio) * y_boosted
        fused /= fused.sum(axis=1, keepdims=True)

        # Prédiction et évaluation
        y_pred_fused = np.argmax(fused, axis=1)

        df_strike_preds = pd.DataFrame({
            'Frame': np.arange(11, 11+len(y_pred_fused)),
            'Strike_Pred': y_pred_fused,
            'Strike_Prob_top' : fused[:, 1],
            'Strike_Prob_bottom' : fused[:, 2],
            'peak_prob': aligned,
            'traj_prob_top': y_pred_proba_GRU_CNN[:, 1],
            'traj_prob_bottom': y_pred_proba_GRU_CNN[:, 2]
        })

        true_frame_indices = df_predict_tracknet.index.to_numpy()

        true_frame_indices = true_frame_indices[11:]
        # Remplacer l'ancienne colonne 'Frame' par les vrais indices
        df_strike_preds["Frame"] = true_frame_indices

        # Réindexer le DataFrame avec la nouvelle colonne
        # Sauvegarder dans outputs_path
        csv_path = os.path.join(outputs_path, "strike_preds.csv")
        df_strike_preds.to_csv(csv_path, index=False)

        print(f"✅ DataFrame des prédictions strike sauvegardé dans : {csv_path}")



        # Sauvegarder le DataFrame
        with open(os.path.join(outputs_path, videoName[:-4] + '_df_predict.pkl'), 'wb') as file:
            pickle.dump(df_predict_tracknet, file)

        

def annotate_and_save_video(video_path, outputs_path, video_name, strike_preds_path, tracknet_pkl_path, reference_path):
    import os
    import cv2
    import pandas as pd
    from camera_changes import detect_similar_camera

    # Charger les prédictions strike
    df_strike_preds = pd.read_csv(strike_preds_path)
    df_strike_preds = df_strike_preds.set_index("Frame")

    # Grouper les frames de strike rapprochées
    frames_strike = df_strike_preds[df_strike_preds['Strike_Pred'] != 0].index.to_list()
    groupes = []
    groupe_courant = []

    for f in frames_strike:
        if not groupe_courant:
            groupe_courant = [f]
        elif f - groupe_courant[-1] <= 7:
            groupe_courant.append(f)
        else:
            groupes.append(groupe_courant)
            groupe_courant = [f]

    if groupe_courant:
        groupes.append(groupe_courant)

    #Choisir un centre par groupe et lui attribuer la classe majoritaire
    cercle_frames = {}  

    centres = []
    for groupe in groupes:
        centre = groupe[len(groupe) // 3]
        centres.append((centre, groupe))  # On garde aussi le groupe associé pour récupérer la classe ensuite

    centres = sorted(centres, key=lambda x: x[0])  # Trie par frame

    # Si une un centre est trop proche avant et après à 20 frames près d'un autre centre alors on ne le garde pas 
    # On fait cela pour réduire le nombre de faux positif
    for i, (centre, groupe) in enumerate(centres):
        prev_frame = centres[i - 1][0] if i > 0 else None
        next_frame = centres[i + 1][0] if i < len(centres) - 1 else None

        too_close_prev = prev_frame is not None and (centre - prev_frame) < 20
        too_close_next = next_frame is not None and (next_frame - centre) < 20

        if not (too_close_prev and too_close_next):  # Garde si au moins un est à plus de 20 frames
            strikes_du_groupe = df_strike_preds.loc[groupe, 'Strike_Pred']
            classe_majoritaire = strikes_du_groupe.mode().iloc[0]
            cercle_frames[centre] = classe_majoritaire



    # Charger les prédictions TrackNet
    with open(tracknet_pkl_path, "rb") as f:
        df_predict_tracknet = pickle.load(f)

    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Détection des scènes similaires
    print("🔍 Détection des scènes similaires (annotate)...")
    similar_scenes = detect_similar_camera(video_path, reference_path)
    print(f"➡️ {len(similar_scenes)} scènes similaires détectées pour annotation.")

    # Configuration de la vidéo de sortie
    output_path = os.path.join(outputs_path, video_name[:-4] + '_annotated6.mp4')
    if os.path.exists(output_path):
        os.remove(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for idx_scene, scene in enumerate(similar_scenes):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        print(f"✏️ Annotation de la scène {idx_scene + 1}: frames {start_frame} → {end_frame}")

        # Positionner la lecture sur la bonne frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame + 1):
            success, frame = cap.read()
            if not success:
                break
            # Annotation du volant. Si le volant est visible, on dessine un petit point rouge à sa position.
            if frame_idx in df_predict_tracknet.index:
                shuttle_visible = df_predict_tracknet.loc[frame_idx, "Visibility_Shuttle"]
                if shuttle_visible == 1:
                    x = int(df_predict_tracknet.loc[frame_idx, "ShuttleX"])
                    y = int(df_predict_tracknet.loc[frame_idx, "ShuttleY"])
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Volant
                
                else:
                    if 'x' in locals() and 'y' in locals():
                        # Reutiliser les coordonnées précédentes
                        pass  # x, y restent les mêmes
                    else:
                        continue  
              
                # Annotation du coup. Si la frame est centrale d’un groupe, on dessine un cercle large vert (top) ou bleu (bottom) autour du volant.
                if frame_idx in df_strike_preds.index and frame_idx in cercle_frames:
                
                    strike = cercle_frames[frame_idx]
                    prob_top = df_strike_preds.loc[frame_idx, "Strike_Prob_top"]
                    prob_bottom = df_strike_preds.loc[frame_idx, "Strike_Prob_bottom"]

                    if strike == 1:
                        cv2.circle(frame, (x, y), 20, (0, 255, 0), 3)  # Top
                    elif strike == 2:
                        cv2.circle(frame, (x, y), 20, (255, 0, 0), 3)  # Bottom

                if frame_idx in df_strike_preds.index:

                    prob_top = df_strike_preds.loc[frame_idx, "Strike_Prob_top"]
                    prob_bottom = df_strike_preds.loc[frame_idx, "Strike_Prob_bottom"]

                    cv2.putText(frame,
                                f"Bottom: {prob_top:.2f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2)

                    cv2.putText(frame,
                                f"top: {prob_bottom:.2f}",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2)


            out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Vidéo annotée sauvegardée : {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete Tracking Script (TrackNetV2 only)')
    parser.add_argument('--inputs_path', type=str, help='Chemin vers le dossier d\'entrée')
    parser.add_argument('--outputs_path', type=str, help='Chemin vers le dossier de sortie')
    args = parser.parse_args()

    if args.inputs_path is None or args.outputs_path is None:
        parser.print_help()
        sys.exit()

    # if not os.path.exists(args.outputs_path):
    #     print("Outputs path does not exist")
    #     sys.exit()
    # if not os.path.exists(args.inputs_path):
    #     print("Inputs path does not exist")
    #     sys.exit()

    #ecriture POUR ONYXIA
        

    def download_s3_to_local(s3_path, local_path):
        import subprocess
        cmd = f"mc cp {s3_path} {local_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Erreur lors du téléchargement de {s3_path} :\n{result.stderr.decode()}")

    import subprocess
    import re

    def is_youtube_url(url):
        return re.match(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/", url) is not None


    if is_youtube_url(args.inputs_path):
        video_url = args.inputs_path
        local_video_path = "/tmp/video.mp4"

        if not os.path.exists(local_video_path):
            print("📥 Téléchargement de la vidéo YouTube avec yt-dlp...")
            command = [
                "yt-dlp",
                "-f", "best[ext=mp4][acodec!=none]/best",
                "-o", local_video_path,
                video_url
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("❌ Erreur yt-dlp:", result.stderr.decode())
                sys.exit(1)
            print("✅ Téléchargement terminé :", local_video_path)

        # Générer l'image de référence automatiquement
        reference_path = "/tmp/reference.jpg"
        extract_reference_frame(local_video_path, reference_path)

        inputs_path = "/tmp"
        videos = ["video.mp4"]
        references = ["reference.jpg"]

    elif args.inputs_path.startswith("s3/"):
        match_filename = os.path.basename(args.inputs_path)
        match_id = match_filename.split('.')[0]

        # Télécharger la vidéo
        local_video_path = f"/tmp/{match_filename}"
        if not os.path.exists(local_video_path):
            download_s3_to_local(args.inputs_path, local_video_path)

        # Télécharger l'image de référence
        reference_s3_path = os.path.join(os.path.dirname(args.inputs_path), f"{match_id}_reference.jpg")
        local_reference_path = f"/tmp/{match_id}_reference.jpg"
        # Essayer de télécharger depuis le S3
        reference_found = False
        if not os.path.exists(local_reference_path):
            try:
                print("🔍 Recherche de l'image de référence sur le S3...")
                download_s3_to_local(reference_s3_path, local_reference_path)
                reference_found = True
                print("🟢 Image de référence trouvée et téléchargée depuis le S3.")
            except RuntimeError as e:
                print("⚠️ Image non trouvée sur le S3 :", e)

        # Si toujours pas disponible, la créer depuis la vidéo
        if not os.path.exists(local_reference_path):
            print("📸 Génération locale de l'image de référence depuis la vidéo...")
            extract_reference_frame(local_video_path, local_reference_path)
        else:
            print("✅ Image de référence prête :", local_reference_path)
        inputs_path = "/tmp"
        videos = [match_filename]
        references = [f"{match_id}_reference.jpg"]

    else:

        if os.path.isfile(args.inputs_path) and args.inputs_path.lower().endswith(".mp4"):

        # Cas: chemin direct vers une vidéo locale

            inputs_path = os.path.dirname(args.inputs_path)

            video_filename = os.path.basename(args.inputs_path)

            videos = [video_filename]
 
        # Générer automatiquement l'image de référence (même logique que plus haut)

            match_id = os.path.splitext(video_filename)[0]

            reference_filename = f"{match_id}_reference.jpg"

            reference_path = os.path.join(inputs_path, reference_filename)
 
            if not os.path.exists(reference_path):

                print("📸 Génération locale de l'image de référence depuis la vidéo...")

                extract_reference_frame(args.inputs_path, reference_path)

            else:

                print("✅ Image de référence déjà présente :", reference_path)
    
            # La fonction complete_tracking attend des noms relatifs au dossier inputs_path

            references = [reference_filename]

        else:

            # Cas: on t’a donné un dossier (comportement inchangé)

            inputs_path = args.inputs_path

            videos = lister_fichiers(inputs_path, ['.mp4'])

            references = lister_fichiers(inputs_path, ['.jpg', '.png'])
    


    complete_tracking(inputs_path, videos, references, args.outputs_path)


    video_originale = os.path.join(inputs_path, videos[0])
    reference_image_path = os.path.join(inputs_path, references[0])
    annotate_and_save_video(
        video_path=video_originale,
        outputs_path=args.outputs_path,
        video_name=videos[0],
        strike_preds_path=os.path.join(args.outputs_path, "strike_preds.csv"),
        tracknet_pkl_path=os.path.join(args.outputs_path, videos[0][:-4] + '_df_predict.pkl'),
        reference_path=reference_image_path
    )   
    print("fin")
