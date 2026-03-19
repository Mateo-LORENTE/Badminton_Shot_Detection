import numpy as np
from skimage.transform import resize

'''
Préparation des données de coordonnées du volant pour inférence du modèle GRU
'''

def pre_processing_for_inference(df_dataset):
    # Renommage des colonnes si besoin
    df_dataset = df_dataset.rename(columns={
        'shuttle_x': 'norm_shuttle_x',
        'shuttle_y': 'norm_shuttle_y',
        'ShuttleX': 'norm_shuttle_x',
        'ShuttleY': 'norm_shuttle_y',
    })

    # Ne garder que les colonnes utiles
    df_dataset = df_dataset[['norm_shuttle_y', 'norm_shuttle_x']].copy()

    eps = 1e-8
    num_consec = 12
    speed = 1.0

    # normalise les coordonnées X/Y entre 0 et 1 (mais ajoute +1 ensuite, ce qui donne une plage [1,2]) en ignorant les valeurs nulles ou très petites
    def scale_data(x):
        x = np.array(x)
        cols = list(range(x.shape[1]))
        def scale_by_col(x, cols):
            x_ = np.array(x[:, cols])
            idx = np.abs(x_) < eps
            m, M = np.min(x_[~idx]), np.max(x_[~idx])
            x_[~idx] = (x_[~idx] - m) / (M - m) + 1
            x[:, cols] = x_
            return x
        return scale_by_col(x, cols)

    # Redimensionne la séquence en longueur avec interpolation (resize), utile pour accélérer ou ralentir le temps (via speed)
    def resample(series, s):
        flatten = False
        if len(series.shape) == 1:
            series.resize((series.shape[0], 1))
            series = np.array(series).astype('float64')
            flatten = True
        if np.all(np.isnan(series)):
            series = np.zeros_like(series)
        series = resize(series, (int(s * series.shape[0]), series.shape[1]))
        if flatten:
            series = series.flatten()
        return series

    # Classe interne permettant de gérer l'interpolation lorsque le volant n'est pas détecté sur des des courtes plages de temps (occlusion, perte de détection…).
    class Trajectory:
        def __init__(self, df_name, interp=True):
            trajectory = df_name.rename(columns={'norm_shuttle_x': 'X', 'norm_shuttle_y': 'Y'})
            if interp:
                trajectory['X'] = trajectory['X'].replace(0, np.nan)
                trajectory['Y'] = trajectory['Y'].replace(0, np.nan)
                for coord in ['X', 'Y']:
                    for i in range(1, 6):
                        mask = (
                            trajectory[coord].isna() &
                            trajectory[coord].shift(-i).isna() &
                            trajectory[coord].shift(-6).notna() &
                            (trajectory[coord].shift(1).notna() if i == 1 else True)
                        )
                        for idx in trajectory[mask].index:
                            trajectory.loc[idx-1:idx+i, coord] = (
                                trajectory.loc[idx-1:idx+i, coord]
                                .interpolate(method='linear', limit=i, limit_direction='forward')
                            )
                trajectory['X'] = trajectory['X'].replace(np.nan, 0)
                trajectory['Y'] = trajectory['Y'].replace(np.nan, 0)
            self.X = trajectory['X'].tolist()
            self.Y = trajectory['Y'].tolist()

    # Générer les trajectoires
    trajectory = Trajectory(df_dataset, interp=True)
    trajectory.X = resample(np.array(trajectory.X), speed)
    trajectory.Y = resample(np.array(trajectory.Y), speed)

    # Construit des fenêtres glissantes de num_consec frames
    x_list = []
    for i in range(num_consec):
        end = len(trajectory.X) - num_consec + i + 1
        x_bird = np.array(list(zip(trajectory.X[i:end], trajectory.Y[i:end])))
        x = np.hstack([x_bird])
        x_list.append(x)

    #Les séquences sont concaténées horizontalement, puis normalisées avant d’être retournées.
    x_t = np.hstack(x_list)
    traj = scale_data(x_t)
    return traj
