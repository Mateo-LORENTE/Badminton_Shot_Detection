import numpy as np
import librosa
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

from scipy.signal import find_peaks
import cv2
import pandas as pd
import scipy.signal
import scipy.stats
import scipy.signal







def peak_extractor(chemin_audio, chemin_video, chemin_df, nom_video):
    """
    Détecte les pics audio haute fréquence dans une vidéo, 
    puis les associe aux événements de type "hit" présents dans un DataFrame d’annotations.

    Étapes principales :
    1. **Extraction audio** :
       - Si `chemin_audio` n’existe pas, extrait l’audio de `chemin_video` et le sauvegarde.
       - Charge l’audio avec `librosa` en conservant la fréquence d’échantillonnage d’origine.

    2. **Prétraitement audio** :
       - Filtre passe-haut (Butterworth) à partir de 4 kHz pour isoler les hautes fréquences.
       - Calcule un spectrogramme mel dans la bande [2 kHz – 10 kHz].
       - Moyenne des bandes au-dessus de 100 bins mel pour obtenir une énergie haute fréquence.
       - Lissage et dérivée de l’énergie pour repérer les variations rapides.

    3. **Détection des pics** :
       - Seuil basé sur la moyenne + écart-type de la dérivée lissée.
       - Utilisation de `scipy.signal.find_peaks` avec contraintes de distance, largeur et proéminence.
       - Conversion des indices de pics en temps (`peak_times`).

    4. **Alignement avec annotations** :
       - Charge `chemin_df` (pickle) contenant les annotations vidéo (`hit` ou `no`).
       - Crée un DataFrame `df_dataset` avec les frames, temps, labels binaires et colonnes de pics (`Peak_binary`, `PEAK_HIT`).
       - Marque un `Peak_binary = 1` pour les frames où un pic est détecté.
       - Définit une fenêtre temporelle (`window_before`, `window_after`) dépendante de `nom_video` ou du FPS, 
         pour associer un pic à un hit.

    5. **Calcul des stats** :
       - Affiche le nombre total de hits, de pics détectés et le pourcentage de hits correctement associés.
       - Affiche également la précision par rapport au nombre total de pics.

    6. **Retour** :
       - `df_peaks` : DataFrame des pics détectés avec leurs frames, temps et si c’est un pic associé à un hit (`PEAK_HIT`).
       - `fps` : nombre d’images par seconde de la vidéo.
       - `sr` : fréquence d’échantillonnage audio.
       - `y` : signal audio brut filtré.

    Paramètres
    ----------
    chemin_audio : str
        Chemin du fichier audio (.wav ou .mp3) associé à la vidéo.
    chemin_video : str
        Chemin du fichier vidéo (.mp4, .avi, etc.).
    chemin_df : str
        Chemin du fichier pickle contenant le DataFrame des annotations vidéo.
    nom_video : str
        Identifiant ou nom de la vidéo, utilisé pour ajuster la fenêtre d’association pics-hits.

    Retour
    ------
    df_peaks : pandas.DataFrame
        Liste des pics audio détectés avec leurs temps et association aux hits.
    fps : float
        Fréquence d’images de la vidéo.
    sr : int
        Fréquence d’échantillonnage audio.
    y : numpy.ndarray
        Signal audio chargé et filtré.
    """

    # Extraction ou chargement audio
    if not os.path.exists(chemin_audio):
        clip = VideoFileClip(chemin_video)
        clip.audio.write_audiofile(chemin_audio)

    y, sr = librosa.load(chemin_audio, sr=None)  
    y = y.astype(np.float32)

    dataset = {
        'Frame': [],
        'Time (s)': [],
        'Hit Binary Label': [],
        'Peak_binary' : [],
        'PEAK_HIT' : [],
        # 'ShuttleX' : [],
        # 'ShuttleY' : [],
        'hit' : []
        
    }

    # === Charger les prédictions ===
    df = pd.read_pickle(chemin_df)



    # === Paramètres vidéo ===
    cap = cv2.VideoCapture(chemin_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Paramètres
    hop_length = 256



    # 1) Passe-haut  high-pass butterworth
    b, a = scipy.signal.butter(4, 4000/(sr/2), btype='high')
    y_hp = scipy.signal.filtfilt(b, a, y)
    y_hp = y_hp.astype(np.float32)
    # 2) Enveloppe HF
    S = librosa.feature.melspectrogram(y=y_hp, sr=sr, n_mels=256,fmin=2000, fmax=10000, hop_length=hop_length,dtype=np.float32)
    S_dB = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    hfe = S_dB[100:, :].mean(axis=0)

    
    # === Calcul de la dérivée lissée sur toute la série ===
    dh  = np.diff(hfe, prepend=hfe[0])
    dhs = scipy.ndimage.gaussian_filter1d(dh, sigma=1)



    # === Seuil global et paramètres de find_peaks ===
    thr = dhs.mean() + 1 * dhs.std()

    # conversion secondes → frames (pour la largeur)
    min_w_sec, max_w_sec = 0.0, 0.12
    min_w_frames = int(min_w_sec * sr / hop_length)  # ici 0
    max_w_frames = int(max_w_sec * sr / hop_length)

    # === Détection des pics sur l’ensemble de la séquence ===
    peaks, props = find_peaks(
        dhs,
        height=thr,
        distance=6,                    # en frames
        width=(min_w_frames, max_w_frames),
        prominence=3
    )

    # # === Conversion en temps ===
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)





    high_freq_energy = S_dB[100:, :].mean(axis=0)
    # Conversion indices → temps
    frames = np.arange(len(high_freq_energy))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # === Boucler sur les index de data uniquement ===
    for frame_idx in df.index:


        

        t = frame_idx / fps
        # Récupérer le hit original
        hit_value = df.loc[frame_idx, 'hit']

        #coordonnée volant
        # shuttlex = df.loc[frame_idx, 'ShuttleX']
        # shuttley = df.loc[frame_idx, 'ShuttleY']

        # === Traduire en label binaire
        if hit_value == 'no':
            label_binary = 0
        elif hit_value in ['top', 'bottom']:
            label_binary = 1
        else:
            continue  # Skip toute autre valeur inattendue



        dataset['Frame'].append(frame_idx)
        dataset['Time (s)'].append(t)
        dataset['Hit Binary Label'].append(label_binary)
        dataset['Peak_binary'].append(0)
        dataset['PEAK_HIT'].append(0)

        # dataset['ShuttleX'].append(shuttlex)
        # dataset['ShuttleY'].append(shuttley)
        dataset['hit'].append(hit_value)






    # Convertir le dataset en DataFrame temporaire pour facilité d’accès
    df_dataset = pd.DataFrame(dataset)
    n_hits = df_dataset[df_dataset['Hit Binary Label'] == 1].shape[0]
    print(f"Nombre de hits : {n_hits}")
    # Associer chaque peak_time à la frame la plus proche
    for pt in peak_times:
        closest_frame = int(round(pt * fps))
        # Vérifier que ce frame existe dans le dataset
        if closest_frame in df_dataset['Frame'].values:
            idx = df_dataset[df_dataset['Frame'] == closest_frame].index[0]
            df_dataset.at[idx, 'Peak_binary'] = 1



    hit_frames = df_dataset[df_dataset['Hit Binary Label'] == 1]['Frame'].values
    peak_frames = df_dataset[df_dataset['Peak_binary'] == 1]['Frame'].values

    '''
    Décalage temporelle calculée à la main pour chaque vidéo en maximisant le taux de Hit détecter dans la fenetre
    '''

    matched_hits = 0
    window_before = 3
    window_after = 3
    if fps < 28 :
        window_before = 3
        window_after = 3 # nombre de frames après le hit

    elif nom_video == "43835"  or nom_video == "63851":
        window_before = 3
        window_after = 3

    elif nom_video == "43695" or nom_video == "47563" or nom_video == "49887" :
        window_before = 2
        window_after = 4

    elif nom_video == "46959" or nom_video == "49766 ":
        window_before = 4
        window_after = 2

    else: 
        window_before = -3
        window_after = 9 

    '''
    Les écarts suivants son ceux calculés par Yoann avec Adobe mais n'obtienne pas de bonne performance (raison inconnue)
    '''

    # if nom_video == "5684" or "37541" or "42354" or "50821":
    #     window_before = 2
    #     window_after = 4

    # if nom_video == "43695" or "43835" or "47563" or "63851":
    #     window_before = 1
    #     window_after = 5

    # if nom_video == "51185":
    #     window_before = 0
    #     window_after = 6

    # if nom_video == "87177" or "37159":
    #     window_before = -2
    #     window_after = 8
    
    # if nom_video == "34289":
    #     window_before = -4
    #     window_after = 10

    # Réinitialise PEAK_HIT
    df_dataset['PEAK_HIT'] = 0

    matched_hits = 0
    for hf in hit_frames:
        # repère tous les pics dans [hf-window_before, hf+window_after]
        candidats = peak_frames[(peak_frames >= hf - window_before) &
                                (peak_frames <= hf + window_after)]
        if len(candidats) > 0:
            matched_hits += 1

            # 1) trouver le candidat le plus proche du hit hf
            deltas = np.abs(candidats - hf)
            pf_closest = candidats[np.argmin(deltas)]

            # 2) marquer ce pic unique dans df_dataset
            mask = df_dataset['Frame'] == pf_closest
            df_dataset.loc[mask, 'PEAK_HIT'] = 1


    total_hits = len(hit_frames)
    total_peaks = len(peak_frames)

    print(f"Nombre total de hits : {total_hits}")
    print(f"Nombre total de pics audio détectés : {total_peaks}")
    print(f"{matched_hits} sur {total_hits} hits ont un pic associé dans +4 frames ({matched_hits / total_hits:.1%})")
    print(f"accuracy = {matched_hits / total_peaks:.1%}")


    df_peaks = df_dataset.loc[
        df_dataset['Peak_binary'] == 1,
        ['Frame', 'Time (s)', 'PEAK_HIT']
    ].reset_index(drop=True)
    

    return df_peaks,fps,sr,y #df_dataset




#------------------------------- FEATURE EXTRACTOR --------------------------------------------------------------------
def features_extractor(df_peaks,fps,sr,y,chemin_sauvegarde_csv):

    """
    Extrait un large ensemble de features audio centrées sur chaque pic détecté,
    les enregistre dans un CSV et retourne le DataFrame résultant.

    Principe :
      - Pour chaque pic (frame, temps, PEAK_HIT) de `df_peaks`, extrait un segment
        audio d’une durée ≈ 1 frame vidéo autour du centre.
      - Calcule des descripteurs temps/fréquence, puis stocke une ligne de features
        par pic.

    Features calculées (principales) :
      - Temps : ZCR, RMS, enveloppe (Attack_Time, Decay_Time), autocorr (1er pic).
      - Spectre : centroid, bandwidth, rolloff, flatness, skewness, kurtosis.
      - Mel-spec : moyenne/écart-type/max, fréquence de pic, énergies de sous-bandes
        (2–4, 4–6, 6–8, 8–10 kHz), score « Shoe_Screech » (bandes ciblées).
      - Dynamique : spectral flux (moyenne, max, std).
      - Représentations : MFCC 1–20 (+ delta, delta²), spectral contrast (1–7),
        tonnetz (1–6), chroma (12).
      - Ondelette : maximum des coefficients CWT (Ricker) sur plusieurs largeurs.

    Effets de bord :
      - Crée le dossier de sortie si nécessaire et écrit `chemin_sauvegarde_csv`.
      - Affiche un classement des corrélations (absolues) de chaque feature
        avec la colonne binaire `PEAK_HIT`.

    Args:
        df_peaks (pd.DataFrame): lignes « pics » avec colonnes ['Frame','Time (s)','PEAK_HIT'].
        fps (float): fréquence d’images de la vidéo (pour convertir frame ↔ temps).
        sr (int): fréquence d’échantillonnage du signal audio.
        y (np.ndarray): signal audio mono aligné à la vidéo.
        chemin_sauvegarde_csv (str): chemin du CSV de sortie.

    Returns:
        pd.DataFrame: tableau des features par pic (une ligne par entrée de `df_peaks`).
    """
    def ricker_wavelet(width):
        
        w = width
        t = np.arange(-w, w+1)
        sigma = w / np.sqrt(2)
        # formule non-normalisée, suffisante pour extraire un max relatif
        wave = (1 - (t**2)/(sigma**2)) * np.exp(-(t**2)/(2*sigma**2))
        return wave


    # Sélection des pics audio seuls


    print(df_peaks.head()) 


    frame_dt      = 1.0 / fps
    # Nombre de samples correspondant à 1 frame
    frame_samps   = int(frame_dt * sr)

    

    # --- Préparation du dictionnaire des features enrichies (extension) ---
    features = {
        'Frame': [], 'Time (s)': [], 'PEAK_HIT': [],
        'ZCR': [], 'RMS': [],
        'Spectral_Centroid': [], 'Spectral_Bandwidth': [],
        'Spectral_Rolloff': [], 'Spectral_Flatness': [],
        'Mean_Spectrogram': [], 'Std_Spectrogram': [], 'Max_Spectrogram': [],
        'Peak_Frequency': [], 'High_Freq_Energy_4k-8kHz': [], 'Shoe_Screech_Score': [],
        # Spectral Flux stats
        'Flux_Mean': [], 'Flux_Max': [], 'Flux_Std': [],
        # Envelope stats
        'Attack_Time': [], 'Decay_Time': [],
        # Skewness & Kurtosis
        'Spec_Skewness': [], 'Spec_Kurtosis': []
    }

    # Ajout des sous-bandes 2–4,4–6,6–8,8–10 kHz
    subbands = [(2000,4000),(4000,6000),(6000,8000),(8000,10000)]
    for lo, hi in subbands:
        features[f'Energy_{lo//1000}-{hi//1000}kHz'] = []

    # Ajout MFCC, Delta, Delta-Delta
    for i in range(1,21):
        features[f'MFCC_{i}'] = []
        features[f'Delta_MFCC_{i}'] = []
        features[f'Delta2_MFCC_{i}'] = []

    # Spectral Contrast (7 bandes) et Tonnetz (6 dim)
    for i in range(1,8):
        features[f'Spectral_Contrast_{i}'] = []
    for i in range(1,7):
        features[f'Tonnetz_{i}'] = []

    # Chroma (12 bins)
    for i in range(1,13):
        features[f'Chroma_{i}'] = []

    # Auto-corrélation
    features['Autocorr_First_Peak_Lag'] = []

    # CWT max coeff
    features['CWT_Max'] = []

    hop_length = 256

    # --- Extraction ---
    for _, row in df_peaks.iterrows():
        frame, t, peak_hit = int(row['Frame']), row['Time (s)'], row['PEAK_HIT']
        center = int(t * sr)
        half = frame_samps
        start, end = max(0, center-half), min(len(y), center+half+1)
        seg = y[start:end]
        if len(seg) < 2: continue

        # ZCR & RMS
        zcr = librosa.feature.zero_crossing_rate(y=seg, frame_length=len(seg), hop_length=len(seg)+1)[0,0]
        rms = librosa.feature.rms(y=seg, frame_length=len(seg), hop_length=len(seg)+1)[0,0]

        # Mel-spectrogram
        S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128, n_fft=1024,
                                        hop_length=len(seg)+1, fmin=50)
        S_dB = librosa.power_to_db(S, ref=np.max)
        mean_spec, std_spec, max_spec = S_dB.mean(), S_dB.std(), S_dB.max()

        # Spectral stats
        centroid  = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=len(seg), hop_length=len(seg)+1)[0].mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr, n_fft=len(seg), hop_length=len(seg)+1)[0].mean()
        rolloff   = librosa.feature.spectral_rolloff(y=seg, sr=sr, n_fft=len(seg), hop_length=len(seg)+1, roll_percent=0.85)[0].mean()
        flatness  = librosa.feature.spectral_flatness(y=seg, n_fft=len(seg), hop_length=len(seg)+1)[0].mean()

        # Peak freq & high-freq energy
        idx = np.unravel_index(np.argmax(S_dB, axis=None), S_dB.shape)
        mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=50)
        peak_freq = mel_freqs[idx[0]]
        hf_mask = (mel_freqs>=4000)&(mel_freqs<=8000)
        hf_energy = S_dB[hf_mask,:].mean()
        shoe_mask = ((mel_freqs>=2800)&(mel_freqs<=3200))|((mel_freqs>=4800)&(mel_freqs<=5200))
        shoe_score = S_dB[shoe_mask,:].mean()

        # Spectral Flux stats
        flux_env = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=hop_length)
        features['Flux_Mean'].append(flux_env.mean())
        features['Flux_Max'].append(flux_env.max())
        features['Flux_Std'].append(flux_env.std())

        # Envelope stats via RMS envelope
        rms_env = librosa.feature.rms(y=seg, frame_length=256, hop_length=256)[0]
        # attack: temps du max RMS relatif au début
        attack = np.argmax(rms_env) * (256/sr)
        # decay: temps de passage sous 1/e du max après le pic
        post = rms_env[np.argmax(rms_env):]
        thresh = rms_env.max()/np.e
        below = np.where(post<thresh)[0]
        decay = (below[0] if below.size else len(post)) * (256/sr)
        features['Attack_Time'].append(attack)
        features['Decay_Time'].append(decay)

        # Skewness & Kurtosis du spectre
        flat = S_dB.flatten()
        features['Spec_Skewness'].append(scipy.stats.skew(flat))
        features['Spec_Kurtosis'].append(scipy.stats.kurtosis(flat))

        # Sub-band energies
        freqs = librosa.mel_frequencies(n_mels=128, fmin=50)
        for lo, hi in subbands:
            mask = (freqs>=lo)&(freqs<hi)
            features[f'Energy_{lo//1000}-{hi//1000}kHz'].append(S_dB[mask,:].mean())

        # MFCC, Delta, Delta-Delta
        hop_mfcc = 256
        raw_mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20, n_fft=1024, hop_length=hop_mfcc)
        mfcc_mean = raw_mfcc.mean(axis=1)
        delta1    = librosa.feature.delta(raw_mfcc).mean(axis=1) if raw_mfcc.shape[1]>1 else np.zeros(20)
        delta2    = librosa.feature.delta(raw_mfcc, order=2).mean(axis=1) if raw_mfcc.shape[1]>2 else np.zeros(20)
        for i,(m,d1,d2) in enumerate(zip(mfcc_mean, delta1, delta2), start=1):
            features[f'MFCC_{i}'].append(m)
            features[f'Delta_MFCC_{i}'].append(d1)
            features[f'Delta2_MFCC_{i}'].append(d2)

        # Spectral Contrast & Tonnetz
        contrast = librosa.feature.spectral_contrast(y=seg, sr=sr, n_fft=1024, hop_length=len(seg)+1).mean(axis=1)
        tonnetz  = librosa.feature.tonnetz(y=seg, sr=sr).mean(axis=1)
        for i,c in enumerate(contrast, start=1):
            features[f'Spectral_Contrast_{i}'].append(c)
        for i,tz in enumerate(tonnetz, start=1):
            features[f'Tonnetz_{i}'].append(tz)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr, n_fft=1024, hop_length=len(seg)+1).mean(axis=1)
        for i,val in enumerate(chroma, start=1):
            features[f'Chroma_{i}'].append(val)

        # Autocorrélation (premier pic hors zéro)
        ac = np.correlate(seg, seg, mode='full')
        mid = len(ac)//2
        post_ac = ac[mid+1:]
        peaks_ac,_ = find_peaks(post_ac)
        lag = peaks_ac[0] if peaks_ac.size else 0
        features['Autocorr_First_Peak_Lag'].append(lag)

        

        # … dans ta boucle d’extraction, remplace l’ancien bloc « CWT » par :
        widths = np.arange(1, 31)
        cwt_max = 0.0
        for w in widths:
            wave = ricker_wavelet(w)
            # convolution « même taille »
            coeff = np.convolve(seg, wave, mode='same')
            cwt_max = max(cwt_max, np.abs(coeff).max())

        features['CWT_Max'].append(cwt_max)

        # Enregistrement des métas
        features['Frame'].append(frame)
        features['Time (s)'].append(t)
        features['PEAK_HIT'].append(peak_hit)
        features['ZCR'].append(zcr)
        features['RMS'].append(rms)
        features['Spectral_Centroid'].append(centroid)
        features['Spectral_Bandwidth'].append(bandwidth)
        features['Spectral_Rolloff'].append(rolloff)
        features['Spectral_Flatness'].append(flatness)
        features['Mean_Spectrogram'].append(mean_spec)
        features['Std_Spectrogram'].append(std_spec)
        features['Max_Spectrogram'].append(max_spec)
        features['Peak_Frequency'].append(peak_freq)
        features['High_Freq_Energy_4k-8kHz'].append(hf_energy)
        features['Shoe_Screech_Score'].append(shoe_score)

    # --- Création du DataFrame final et sauvegarde ---
    df_peak_features = pd.DataFrame(features)
    os.makedirs(os.path.dirname(chemin_sauvegarde_csv), exist_ok=True)
    df_peak_features.to_csv(chemin_sauvegarde_csv, index=False)

    print(f"✅ Dataset final avec features enrichies sauvegardé : {chemin_sauvegarde_csv}")


    # --- Test de corrélation entre PEAK_HIT et les features ---
    # Calcul de la matrice de corrélation
    corr_matrix = df_peak_features.corr()

    # Extraction et tri des corrélations par rapport à PEAK_HIT
    corr_with_peak = (
        corr_matrix['PEAK_HIT']
        .drop('PEAK_HIT')
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )

    # Construction d'un DataFrame pour affichage
    feature_scores = pd.DataFrame({
        'Feature': corr_with_peak.index,
        'Correlation_with_PEAK_HIT': corr_with_peak.values
    })

    print("🎯 Classement des variables par corrélation absolue avec PEAK_HIT :")
    print(feature_scores.to_string(index=False))

    return df_peak_features





def spectre_extractor(y,sr,df_peaks,save_dir,nom_video):
    """
    Extrait et sauvegarde les spectrogrammes log-Mel centrés sur chaque frame 
    indiquée dans `df_peaks`, ainsi que leurs labels associés.

    Pour chaque frame listée :
      - Extrait un segment audio autour du temps correspondant (± 2 frames vidéo à 25 fps).
      - Calcule un spectrogramme Mel (128 bandes, FFT=1024, hop=256, fmin=50 Hz).
      - Convertit en dB (log-Mel) et cast en float32.
      - Associe le label binaire `PEAK_HIT` issu de `df_peaks`.

    Sauvegardes effectuées :
      - Fichier `.npy` des spectrogrammes : `mel_specs_{nom_video}.npy`
        (forme : (N, n_frames, 128)).
      - Fichier `.npy` des labels binaires : `labels_{nom_video}.npy` (forme : (N,)).

    Args:
        y (np.ndarray): signal audio mono aligné à la vidéo.
        sr (int): fréquence d’échantillonnage de `y`.
        df_peaks (pd.DataFrame): contient les frames et labels (`Frame`, `Time (s)`, `PEAK_HIT`).
        save_dir (str): dossier de sortie pour les fichiers `.npy`.
        nom_video (str): identifiant/nom de la vidéo (utilisé dans les noms de fichiers).

    Returns:
        int: 0 si l’extraction et la sauvegarde ont été réalisées avec succès.
    """

    # listes pour stocker les spectrogrammes et les labels
    mel_specs = []
    labels    = []
    
    frame_dt    = 2 / 25
    frame_samps = int(frame_dt * sr)

    for frame_idx in df_peaks['Frame'].values:
        # récupérer le time et le label de la frame
        row       = df_peaks.loc[df_peaks['Frame'] == frame_idx].iloc[0]
        t         = row['Time (s)']
        label_hit = row['PEAK_HIT']        # 0 ou 1

        # extraire le segment audio
        center  = int(t * sr)
        start   = max(0, center - frame_samps)
        end     = min(len(y), center + frame_samps + 1)
        segment = y[start:end]

        # calcul du spectrogramme log-Mel
        S  = librosa.feature.melspectrogram(
                    y=segment, sr=sr,
                    n_mels=128, n_fft=1024,
                    hop_length=256, fmin=50
                )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # on cast en float32 et on ajoute à la liste
        mel_specs.append(S_dB.astype(np.float32))
        labels.append(np.uint8(label_hit))

    # conversion en tableaux NumPy
    mel_specs_array = np.stack(mel_specs, axis=0)  # shape (N, frames, n_mels)
    labels_array    = np.array(labels, dtype=np.uint8)  # shape (N,)

    os.makedirs(save_dir, exist_ok=True)

    # construction des chemins complets
    spec_path  = os.path.join(save_dir, f"mel_specs_{nom_video}.npy")
    label_path = os.path.join(save_dir, f"labels_{nom_video}.npy")

    # sauvegarde
    np.save(spec_path,  mel_specs_array)
    np.save(label_path, labels_array)

    print(f"✅ Enregistré {mel_specs_array.shape[0]} spectrogrammes dans '{spec_path}'")
    print(f"✅ Enregistré {labels_array.size} labels dans '{label_path}'")
    return 0
