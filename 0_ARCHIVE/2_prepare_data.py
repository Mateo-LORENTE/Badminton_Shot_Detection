from extract_players import *
from match_hits import *
import cv2

import argparse
import sys
import os

def lister_fichiers(repertoire, extensions):
	fichiers = []
	for dossier, sous_dossiers, fichiers_dans_dossier in os.walk(repertoire):
		for fichier in fichiers_dans_dossier:
			nom, extension = os.path.splitext(fichier)
			if extension.lower() in extensions:
				fichiers.append(nom+extension)
	fichiers.sort()

	return fichiers

def get_coordinates_dataframe(videoId,height,width,outputs_path):
	# Recharger le dataframe avec les tableaux d'origine depuis le fichier
    with open(outputs_path+'/'+videoId+'_df_predict_with_players_and_hits.pkl', 'rb') as file:
        df_predict_with_players = pickle.load(file)

    df_coordinates = pd.DataFrame(columns=['Frame',
                                        'court_top_left_x', 'court_top_left_y', 'court_top_right_x', 'court_top_right_y',
                                        'court_bottom_left_x', 'court_bottom_left_y', 'court_bottom_right_x', 'court_bottom_right_y',

                                        'shuttle_x', 'shuttle_y',

                                        "nose_top_x", "nose_top_y",
                                        "left_eye_top_x", "left_eye_top_y", 
                                        "right_eye_top_x", "right_eye_top_y",
                                        "left_ear_top_x", "left_ear_top_y",
                                        "right_ear_top_x", "right_ear_top_y",
                                        'left_shoulder_top_x', 'left_shoulder_top_y',
                                        'right_shoulder_top_x', 'right_shoulder_top_y',
                                        'left_elbow_top_x', 'left_elbow_top_y',
                                        'right_elbow_top_x', 'right_elbow_top_y',
                                        'left_wrist_top_x', 'left_wrist_top_y',
                                        'right_wrist_top_x', 'right_wrist_top_y',
                                        'left_hip_top_x', 'left_hip_top_y',
                                        'right_hip_top_x', 'right_hip_top_y',
                                        'left_knee_top_x', 'left_knee_top_y',
                                        'right_knee_top_x', 'right_knee_top_y',
                                        'left_ankle_top_x', 'left_ankle_top_y',
                                        'right_ankle_top_x', 'right_ankle_top_y',

                                        "nose_bottom_x", "nose_bottom_y",
                                        "left_eye_bottom_x", "left_eye_bottom_y",
                                        "right_eye_bottom_x", "right_eye_bottom_y",
                                        "left_ear_bottom_x", "left_ear_bottom_y",
                                        "right_ear_bottom_x", "right_ear_bottom_y",
                                        'left_shoulder_bottom_x', 'left_shoulder_bottom_y',
                                        'right_shoulder_bottom_x', 'right_shoulder_bottom_y',
                                        'left_elbow_bottom_x', 'left_elbow_bottom_y',
                                        'right_elbow_bottom_x', 'right_elbow_bottom_y',
                                        'left_wrist_bottom_x', 'left_wrist_bottom_y',
                                        'right_wrist_bottom_x', 'right_wrist_bottom_y',
                                        'left_hip_bottom_x', 'left_hip_bottom_y',
                                        'right_hip_bottom_x', 'right_hip_bottom_y',
                                        'left_knee_bottom_x', 'left_knee_bottom_y',
                                        'right_knee_bottom_x', 'right_knee_bottom_y',
                                        'left_ankle_bottom_x', 'left_ankle_bottom_y',
                                        'right_ankle_bottom_x', 'right_ankle_bottom_y',

                                        'hit'])

    # Court
    court_coordinates = df_predict_with_players['court_coordinates'].iloc[0]

    for i, row in df_predict_with_players.iterrows():

        # Shuttle
        shuttle_x = row['ShuttleX']
        shuttle_y = row['ShuttleY']

        # Pose Player top
        keypoints_top = row['keypoints_top_player']

        nose_top, left_eye_top, right_eye_top, left_ear_top, right_ear_top, left_shoulder_top, right_shoulder_top, left_elbow_top, right_elbow_top, left_wrist_top, right_wrist_top, \
        left_hip_top, right_hip_top, left_knee_top, right_knee_top, left_ankle_top, right_ankle_top = keypoints_top#[:]

        nose_top_y, nose_top_x, _ = nose_top
        left_eye_top_y, left_eye_top_x, _ = left_eye_top
        right_eye_top_y, right_eye_top_x, _ = right_eye_top
        left_ear_top_y, left_ear_top_x, _ = left_ear_top
        right_ear_top_y, right_ear_top_x, _ = right_ear_top

        left_shoulder_top_y, left_shoulder_top_x, _ = left_shoulder_top
        right_shoulder_top_y, right_shoulder_top_x, _ = right_shoulder_top
        left_elbow_top_y, left_elbow_top_x, _ = left_elbow_top
        right_elbow_top_y, right_elbow_top_x, _ = right_elbow_top
        left_wrist_top_y, left_wrist_top_x, _ = left_wrist_top
        right_wrist_top_y, right_wrist_top_x, _ = right_wrist_top
        left_hip_top_y, left_hip_top_x, _ = left_hip_top
        right_hip_top_y, right_hip_top_x, _ = right_hip_top
        left_knee_top_y, left_knee_top_x, _ = left_knee_top
        right_knee_top_y, right_knee_top_x, _ = right_knee_top
        left_ankle_top_y, left_ankle_top_x, _ = left_ankle_top
        right_ankle_top_y, right_ankle_top_x, _ = right_ankle_top

        # Pose Player bottom
        keypoints_bottom = row['keypoints_bottom_player']

        nose_bottom, left_eye_bottom, right_eye_bottom, left_ear_bottom, right_ear_bottom, left_shoulder_bottom, right_shoulder_bottom, left_elbow_bottom, right_elbow_bottom, left_wrist_bottom, right_wrist_bottom, \
        left_hip_bottom, right_hip_bottom, left_knee_bottom, right_knee_bottom, left_ankle_bottom, right_ankle_bottom = keypoints_bottom#[:]


        nose_bottom_y, nose_bottom_x, _ = nose_bottom
        left_eye_bottom_y, left_eye_bottom_x, _ = left_eye_bottom
        right_eye_bottom_y, right_eye_bottom_x, _ = right_eye_bottom
        left_ear_bottom_y, left_ear_bottom_x, _ = left_ear_bottom
        right_ear_bottom_y, right_ear_bottom_x, _ = right_ear_bottom

        left_shoulder_bottom_y, left_shoulder_bottom_x, _ = left_shoulder_bottom
        right_shoulder_bottom_y, right_shoulder_bottom_x, _ = right_shoulder_bottom
        left_elbow_bottom_y, left_elbow_bottom_x, _ = left_elbow_bottom
        right_elbow_bottom_y, right_elbow_bottom_x, _ = right_elbow_bottom
        left_wrist_bottom_y, left_wrist_bottom_x, _ = left_wrist_bottom
        right_wrist_bottom_y, right_wrist_bottom_x, _ = right_wrist_bottom
        left_hip_bottom_y, left_hip_bottom_x, _ = left_hip_bottom
        right_hip_bottom_y, right_hip_bottom_x, _ = right_hip_bottom
        left_knee_bottom_y, left_knee_bottom_x, _ = left_knee_bottom
        right_knee_bottom_y, right_knee_bottom_x, _ = right_knee_bottom
        left_ankle_bottom_y, left_ankle_bottom_x, _ = left_ankle_bottom
        right_ankle_bottom_y, right_ankle_bottom_x, _ = right_ankle_bottom

        # Hit
        hit = row['Hit']

        df_coordinates.loc[len(df_coordinates)] = [i, court_coordinates[0][0], court_coordinates[0][1],
                                                court_coordinates[1][0], court_coordinates[1][1],
                                                court_coordinates[2][0], court_coordinates[2][1],
                                                court_coordinates[3][0], court_coordinates[3][1],

                                                shuttle_x, shuttle_y,

                                                nose_top_x, nose_top_y,
                                                left_eye_top_x, left_eye_top_y, 
                                                right_eye_top_x, right_eye_top_y,
                                                left_ear_top_x, left_ear_top_y,
                                                right_ear_top_x, right_ear_top_y,

                                                left_shoulder_top_x, left_shoulder_top_y,
                                                right_shoulder_top_x, right_shoulder_top_y,
                                                left_elbow_top_x, left_elbow_top_y,
                                                right_elbow_top_x, right_elbow_top_y,
                                                left_wrist_top_x, left_wrist_top_y,
                                                right_wrist_top_x, right_wrist_top_y,
                                                left_hip_top_x, left_hip_top_y,
                                                right_hip_top_x, right_hip_top_y,
                                                left_knee_top_x, left_knee_top_y,
                                                right_knee_top_x, right_knee_top_y,
                                                left_ankle_top_x, left_ankle_top_y,
                                                right_ankle_top_x, right_ankle_top_y,

                                                nose_bottom_x, nose_bottom_y,
                                                left_eye_bottom_x, left_eye_bottom_y,
                                                right_eye_bottom_x, right_eye_bottom_y,
                                                left_ear_bottom_x, left_ear_bottom_y,
                                                right_ear_bottom_x, right_ear_bottom_y,

                                                left_shoulder_bottom_x, left_shoulder_bottom_y,
                                                right_shoulder_bottom_x, right_shoulder_bottom_y,
                                                left_elbow_bottom_x, left_elbow_bottom_y,
                                                right_elbow_bottom_x, right_elbow_bottom_y,
                                                left_wrist_bottom_x, left_wrist_bottom_y,
                                                right_wrist_bottom_x, right_wrist_bottom_y,
                                                left_hip_bottom_x, left_hip_bottom_y,
                                                right_hip_bottom_x, right_hip_bottom_y,
                                                left_knee_bottom_x, left_knee_bottom_y,
                                                right_knee_bottom_x, right_knee_bottom_y,
                                                left_ankle_bottom_x, left_ankle_bottom_y,
                                                right_ankle_bottom_x, right_ankle_bottom_y,
                                                hit]
        
    df_coordinates.set_index('Frame', inplace=True)

    # Sauvegarder le dataframe avec les tableaux d'origine à l'aide de Pickle
    with open(outputs_path+'/'+videoId+'_df_predict_with_coordinates.pkl', 'wb') as file:
        pickle.dump(df_coordinates, file)

    return

#Fonction pour normaliser toutes les coordonnées en pixel entre [1,2] et 0 si donnée manquante
def normalize_data(videoId,height,width,outputs_path):
    # Recharger le dataframe avec les tableaux d'origine depuis le fichier
    with open(outputs_path+'/'+videoId+'_df_predict_with_players_and_hits.pkl', 'rb') as file:
        df_predict_with_players = pickle.load(file)

    df_norm_coordinates = pd.DataFrame(columns = 
            ['Frame',
            'norm_court_top_left_x', 'norm_court_top_left_y', 'norm_court_top_right_x', 'norm_court_top_right_y',
            'norm_court_bottom_left_x', 'norm_court_bottom_left_y', 'norm_court_bottom_right_x', 'norm_court_bottom_right_y',
            'norm_shuttle_x', 'norm_shuttle_y',
            'norm_left_shoulder_top_x', 'norm_left_shoulder_top_y',
            'norm_right_shoulder_top_x', 'norm_right_shoulder_top_y',
            'norm_left_elbow_top_x', 'norm_left_elbow_top_y',
            'norm_right_elbow_top_x', 'norm_right_elbow_top_y',
            'norm_left_wrist_top_x', 'norm_left_wrist_top_y',
            'norm_right_wrist_top_x', 'norm_right_wrist_top_y',
            'norm_left_hip_top_x', 'norm_left_hip_top_y',
            'norm_right_hip_top_x', 'norm_right_hip_top_y',
            'norm_left_knee_top_x', 'norm_left_knee_top_y',
            'norm_right_knee_top_x', 'norm_right_knee_top_y',
            'norm_left_ankle_top_x', 'norm_left_ankle_top_y',
            'norm_right_ankle_top_x', 'norm_right_ankle_top_y',
            'norm_left_shoulder_bottom_x', 'norm_left_shoulder_bottom_y',
            'norm_right_shoulder_bottom_x', 'norm_right_shoulder_bottom_y',
            'norm_left_elbow_bottom_x', 'norm_left_elbow_bottom_y',
            'norm_right_elbow_bottom_x', 'norm_right_elbow_bottom_y',
            'norm_left_wrist_bottom_x', 'norm_left_wrist_bottom_y',
            'norm_right_wrist_bottom_x', 'norm_right_wrist_bottom_y',
            'norm_left_hip_bottom_x', 'norm_left_hip_bottom_y',
            'norm_right_hip_bottom_x', 'norm_right_hip_bottom_y',
            'norm_left_knee_bottom_x', 'norm_left_knee_bottom_y',
            'norm_right_knee_bottom_x', 'norm_right_knee_bottom_y',
            'norm_left_ankle_bottom_x', 'norm_left_ankle_bottom_y',
            'norm_right_ankle_bottom_x', 'norm_right_ankle_bottom_y',
            'hit'])
        

    #Court
    court_coordinates = df_predict_with_players['court_coordinates'].iloc[0]

    norm_court_top_left_x = 1 + (court_coordinates[0][0]/width)
    norm_court_top_left_y = 1 + (court_coordinates[0][1]/height)
    norm_court_top_right_x = 1 + (court_coordinates[1][0]/width)
    norm_court_top_right_y = 1 + (court_coordinates[1][1]/height)
    norm_court_bottom_left_x = 1 + (court_coordinates[2][0]/width)
    norm_court_bottom_left_y = 1 + (court_coordinates[2][1]/height)
    norm_court_bottom_right_x = 1 + (court_coordinates[3][0]/width)
    norm_court_bottom_right_y = 1 + (court_coordinates[3][1]/height)

    for i, row in df_predict_with_players.iterrows():
        
        #Shuttle
        norm_shuttle_x = 1 + (row['ShuttleX']/width)
        norm_shuttle_y = 1 + (row['ShuttleY']/height)
        
        #Pose Player top
        keypoints_top = row['keypoints_top_player']

        left_shoulder_top, right_shoulder_top, left_elbow_top, right_elbow_top, left_wrist_top, right_wrist_top, \
        left_hip_top, right_hip_top, left_knee_top, right_knee_top, left_ankle_top, right_ankle_top = keypoints_top[5:]

        norm_left_shoulder_top_y, norm_left_shoulder_top_x, _ = [1 + value for value in left_shoulder_top]
        norm_right_shoulder_top_y, norm_right_shoulder_top_x, _ = [1 + value for value in right_shoulder_top]
        norm_left_elbow_top_y, norm_left_elbow_top_x, _ = [1 + value for value in left_elbow_top]
        norm_right_elbow_top_y, norm_right_elbow_top_x, _ = [1 + value for value in right_elbow_top]
        norm_left_wrist_top_y, norm_left_wrist_top_x, _ = [1 + value for value in left_wrist_top]
        norm_right_wrist_top_y, norm_right_wrist_top_x, _ = [1 + value for value in right_wrist_top]
        norm_left_hip_top_y, norm_left_hip_top_x, _ = [1 + value for value in left_hip_top]
        norm_right_hip_top_y, norm_right_hip_top_x, _ = [1 + value for value in right_hip_top]
        norm_left_knee_top_y, norm_left_knee_top_x, _ = [1 + value for value in left_knee_top]
        norm_right_knee_top_y, norm_right_knee_top_x, _ = [1 + value for value in right_knee_top]
        norm_left_ankle_top_y, norm_left_ankle_top_x, _ = [1 + value for value in left_ankle_top]
        norm_right_ankle_top_y, norm_right_ankle_top_x, _ = [1 + value for value in right_ankle_top]
        
        #Pose Player bottom
        keypoints_bottom = row['keypoints_bottom_player']

        left_shoulder_bottom, right_shoulder_bottom, left_elbow_bottom, right_elbow_bottom, left_wrist_bottom, right_wrist_bottom, \
        left_hip_bottom, right_hip_bottom, left_knee_bottom, right_knee_bottom, left_ankle_bottom, right_ankle_bottom = keypoints_bottom[5:]

        norm_left_shoulder_bottom_y, norm_left_shoulder_bottom_x, _ = [1 + value for value in left_shoulder_bottom]
        norm_right_shoulder_bottom_y, norm_right_shoulder_bottom_x, _ = [1 + value for value in right_shoulder_bottom]
        norm_left_elbow_bottom_y, norm_left_elbow_bottom_x, _ = [1 + value for value in left_elbow_bottom]
        norm_right_elbow_bottom_y, norm_right_elbow_bottom_x, _ = [1 + value for value in right_elbow_bottom]
        norm_left_wrist_bottom_y, norm_left_wrist_bottom_x, _ = [1 + value for value in left_wrist_bottom]
        norm_right_wrist_bottom_y, norm_right_wrist_bottom_x, _ = [1 + value for value in right_wrist_bottom]
        norm_left_hip_bottom_y, norm_left_hip_bottom_x, _ = [1 + value for value in left_hip_bottom]
        norm_right_hip_bottom_y, norm_right_hip_bottom_x, _ = [1 + value for value in right_hip_bottom]
        norm_left_knee_bottom_y, norm_left_knee_bottom_x, _ = [1 + value for value in left_knee_bottom]
        norm_right_knee_bottom_y, norm_right_knee_bottom_x, _ = [1 + value for value in right_knee_bottom]
        norm_left_ankle_bottom_y, norm_left_ankle_bottom_x, _ = [1 + value for value in left_ankle_bottom]
        norm_right_ankle_bottom_y, norm_right_ankle_bottom_x, _ = [1 + value for value in right_ankle_bottom]
        
        #Hit
        hit = row['Hit']
        
        df_norm_coordinates.loc[len(df_norm_coordinates)] = [i,norm_court_top_left_x, norm_court_top_left_y, norm_court_top_right_x, norm_court_top_right_y, norm_court_bottom_left_x, norm_court_bottom_left_y, norm_court_bottom_right_x, norm_court_bottom_right_y, norm_shuttle_x, norm_shuttle_y, norm_left_shoulder_top_x, norm_left_shoulder_top_y, norm_right_shoulder_top_x, norm_right_shoulder_top_y, norm_left_elbow_top_x, norm_left_elbow_top_y, norm_right_elbow_top_x, norm_right_elbow_top_y, norm_left_wrist_top_x, norm_left_wrist_top_y, norm_right_wrist_top_x, norm_right_wrist_top_y, norm_left_hip_top_x, norm_left_hip_top_y, norm_right_hip_top_x, norm_right_hip_top_y, norm_left_knee_top_x, norm_left_knee_top_y, norm_right_knee_top_x, norm_right_knee_top_y, norm_left_ankle_top_x, norm_left_ankle_top_y, norm_right_ankle_top_x, norm_right_ankle_top_y, norm_left_shoulder_bottom_x, norm_left_shoulder_bottom_y, norm_right_shoulder_bottom_x, norm_right_shoulder_bottom_y, norm_left_elbow_bottom_x, norm_left_elbow_bottom_y, norm_right_elbow_bottom_x, norm_right_elbow_bottom_y, norm_left_wrist_bottom_x, norm_left_wrist_bottom_y, norm_right_wrist_bottom_x, norm_right_wrist_bottom_y, norm_left_hip_bottom_x, norm_left_hip_bottom_y, norm_right_hip_bottom_x, norm_right_hip_bottom_y, norm_left_knee_bottom_x, norm_left_knee_bottom_y, norm_right_knee_bottom_x, norm_right_knee_bottom_y, norm_left_ankle_bottom_x, norm_left_ankle_bottom_y, norm_right_ankle_bottom_x, norm_right_ankle_bottom_y, hit]

    df_norm_coordinates.set_index('Frame',inplace=True)

    df_norm_coordinates.replace(1, 0, inplace=True)

    # Sauvegarder le dataframe avec les tableaux d'origine à l'aide de Pickle
    with open(outputs_path+'/'+videoId+'_df_predict_normalized.pkl', 'wb') as file:
        pickle.dump(df_norm_coordinates, file)

    return

def prepare_data(inputs_path, videos, references, court_coordinates, outputs_path):
    for videoName in videos:
        cap = cv2.VideoCapture(inputs_path+'/'+videoName)
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()
        cv2.destroyAllWindows()

        videoId = videoName[:-4]
        #extract_players(videoId,inputs_path,outputs_path, court_coordinates[0])

        #Attention j'ai changé le nom du fichier dans la fonction match_hits
        match_hits(videoId,fps,inputs_path,outputs_path)

        #normalize_data(videoId,height,width,outputs_path)
        #get_coordinates_dataframe(videoId,height,width,outputs_path)

if __name__ == '__main__':
	# Création de l'objet ArgumentParser
	parser = argparse.ArgumentParser(description='Complete Tracking Script')

	# Ajout des arguments
	parser.add_argument('--inputs_path', type=str, help='Chemin vers le dossier d\'entrée')
	parser.add_argument('--outputs_path', type=str, help='Chemin vers le dossier de sortie')

	# Récupération des arguments
	args = parser.parse_args()

	# Vérification des arguments inputs_path et outputs_path
	if args.inputs_path is None or args.outputs_path is None:
		parser.print_help()
		sys.exit()

	#Vérification des chemins
	if not os.path.exists(args.outputs_path):
		print("Outputs path does not exist")
		sys.exit()
	if not os.path.exists(args.inputs_path):
		print("Inputs path does not exist")
		sys.exit()

	#Listing des fichiers d'entrée
	videos = lister_fichiers(args.inputs_path,['.mp4'])
	references = lister_fichiers(args.inputs_path,['.jpg','.png'])
	csv_court_coordinates = lister_fichiers(args.inputs_path,['.csv','.xlsx'])
	
    
    # Appel de la fonction complete_tracking avec les arguments fournis
	prepare_data(args.inputs_path,videos,references,csv_court_coordinates, args.outputs_path)
	

        

