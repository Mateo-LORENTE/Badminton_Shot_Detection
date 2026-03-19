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
from tensorflow.keras.preprocessing.sequence import pad_sequences




'''
Détecter les pics audio haute fréquence associés aux impacts du volant dans une vidéo, et retourner :
-Un DataFrame avec les frames et l’indication binaire de "pic"
-Un sous-ensemble avec les seuls pics détectés
-Et les données audio utiles (fps, sr, y)
'''


def peak_extractor(chemin_audio, chemin_video, chemin_df):
    # Extraction ou chargement audio
    if not os.path.exists(chemin_audio):
        clip = VideoFileClip(chemin_video)
        clip.audio.write_audiofile(chemin_audio)

    y, sr = librosa.load(chemin_audio, sr=None)  
    y = y.astype(np.float32)

    dataset = {
        'Frame': [],
        'Time (s)': [],
        'Peak_binary' : [],
        'ShuttleX' : [],
        'ShuttleY' : []
        
    }

    # === Charger les prédictions ===
    df = chemin_df



    # === Paramètres vidéo ===
    cap = cv2.VideoCapture(chemin_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Paramètres
    hop_length = 256



    # 1) Passe-haut  high-pass butterworth pour isoler les sons d’impact du volant.
    b, a = scipy.signal.butter(4, 4000/(sr/2), btype='high')
    y_hp = scipy.signal.filtfilt(b, a, y)
    y_hp = y_hp.astype(np.float32)
    # 2) Calcul de l’énergie moyenne haute fréquence à partir du spectrogramme Mel.
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

        shuttlex = df.loc[frame_idx, 'ShuttleX']
        shuttley = df.loc[frame_idx, 'ShuttleY']
        dataset['Frame'].append(frame_idx)
        dataset['Time (s)'].append(t)
        dataset['Peak_binary'].append(0)
        dataset['ShuttleX'].append(shuttlex)
        dataset['ShuttleY'].append(shuttley)
    
    # Convertir le dataset en DataFrame temporaire pour facilité d’accès
    df_dataset = pd.DataFrame(dataset)

     # Associer chaque peak_time à la frame la plus proche
    for pt in peak_times:
        closest_frame = int(round(pt * fps))
        # Vérifier que ce frame existe dans le dataset
        if closest_frame in df_dataset['Frame'].values:
            idx = df_dataset[df_dataset['Frame'] == closest_frame].index[0]
            df_dataset.at[idx, 'Peak_binary'] = 1

    # Création du dataset avec uniquement les pics pour ajout de features
    df_peaks = df_dataset.loc[
        df_dataset['Peak_binary'] == 1,
        ['Frame', 'Time (s)']
    ].reset_index(drop=True)
    
    

    return df_dataset, df_peaks,fps,sr,y



'''
Extraire descripteurs audio riches autour de chaque pic détecté (frame)
'''

#------------------------------- FEATURE EXTRACTOR --------------------------------------------------------------------
def features_extractor(df_peaks,fps,sr,y,chemin_sauvegarde_csv):
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
        'Frame': [], 'Time (s)': [],
        'ZCR': [], 'RMS': [],
        'Spectral_Centroid': [], 'Spectral_Bandwidth': [],
        'Spectral_Rolloff': [], 'Spectral_Flatness': [],
        'Mean_Spectrogram': [], 'Std_Spectrogram': [], 'Max_Spectrogram': [],
        'Peak_Frequency': [], 'High_Freq_Energy_4k-8kHz': [], 'Shoe_Screech_Score': [],
        'Flux_Mean': [], 'Flux_Max': [], 'Flux_Std': [],
        'Attack_Time': [], 'Decay_Time': [],
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
        frame, t = int(row['Frame']), row['Time (s)']
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
        width = min(9, raw_mfcc.shape[1] // 2 * 2 + 1)
        delta1 = librosa.feature.delta(raw_mfcc, width=width).mean(axis=1)
        delta2 = librosa.feature.delta(raw_mfcc, order=2, width=width).mean(axis=1)
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
    save_path = os.path.join(os.path.dirname(chemin_sauvegarde_csv), "feats.csv")
    df_peak_features.to_csv(save_path, index=False)
    print(f"✅ Dataset final avec features enrichies sauvegardé : {chemin_sauvegarde_csv}")





'''
Extraire les spectrogrammes log-Mel autour des pics
'''
def spectre_extractor(y,sr,df_peaks,save_dir):
    
    # listes pour stocker les spectrogrammes et les labels
    mel_specs = []
    labels    = []
    
    frame_dt    = 2 / 25
    frame_samps = int(frame_dt * sr)

    for frame_idx in df_peaks['Frame'].values:
        # récupérer le time et le label de la frame
        row       = df_peaks.loc[df_peaks['Frame'] == frame_idx].iloc[0]
        t         = row['Time (s)']

        # extraire le segment audio
        center  = int(t * sr)
        start   = max(0, center - frame_samps)
        end     = min(len(y), center + frame_samps + 1)
        if frame_idx == 1:
            start = 0
            end = 2*frame_samps + 1
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

    # conversion en tableaux NumPy
    # Extrait les tailles max
    max_frames = max([s.shape[0] for s in mel_specs])
    n_mels = mel_specs[0].shape[1]

    # Remplissage à droite avec des zéros
    mel_specs_padded = [np.pad(s, ((0, max_frames - s.shape[0]), (0, 0)), mode='constant') for s in mel_specs]
    mel_specs_array = np.stack(mel_specs_padded, axis=0)

    os.makedirs(save_dir, exist_ok=True)

    # construction des chemins complets
    spec_path  = os.path.join(save_dir, f"mel_specs.npy")
   
    # sauvegarde
    np.save(spec_path,  mel_specs_array)
    print(f"✅ Enregistré {mel_specs_array.shape[0]} spectrogrammes dans '{spec_path}'")
    return 0
