from run_mmpose import *
from camera_changes import *
import pandas as pd
import pickle

import argparse
import os
import sys
import time

def lister_fichiers(repertoire, extensions):
	fichiers = []
	for dossier, sous_dossiers, fichiers_dans_dossier in os.walk(repertoire):
		for fichier in fichiers_dans_dossier:
			nom, extension = os.path.splitext(fichier)
			if extension.lower() in extensions:
				fichiers.append(nom+extension)
	fichiers.sort()

	return fichiers

def complete_tracking(inputs_path, videos,references,csv_court_coordinates, outputs_path):
    for videoName,reference in zip(videos,references):
        if videoName[:-4] not in reference:
            print('Error ! Video '+videoName+' does not match with reference '+reference)
            continue

        video_path = inputs_path+'/'+videoName
        reference_frame_path = inputs_path+'/'+reference

        #Détection des scènes avec angle classique
        print("Beginning detecting similar scenes")
        similar_scenes = detect_similar_camera(video_path,reference_frame_path)
        print("Done...")

        #On ouvre la vidéo: 
        cap = cv2.VideoCapture(video_path)
        ret, image1 = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(outputs_path+'/'+videoName[:-4]+'_predict.mp4', fourcc, fps, (1920,1080))

        print('Beginning predicting......')

        start = time.time()

        #On définit un dataframe que l'on va sauvegarder
        df_predict_mmpose = pd.DataFrame(columns=['Frame','Keypoints_with_scores'])
        df_predict_mmpose.set_index('Frame')

        #On parcourt les scènes qui nous intéressent
        for scene in similar_scenes:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()

            # Set the video frame position to the start of the scene
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Process frames within the scene
            for frame_number in range(start_frame, end_frame + 1):
                success, image = cap.read()
            
                ### Estimation des positions ###
                # Resize image
                img = image.copy()
                image = run_mmpose(img, frame_number, det_model, pose_model, visualizer, df_predict_mmpose)

                out.write(image)

        df_predict_mmpose = df_predict_mmpose[~df_predict_mmpose.index.duplicated(keep='first')]

        # Sauvegarder le dataframe avec les tableaux d'origine à l'aide de Pickle
        with open(outputs_path+'/'+videoName[:-4]+'_df_mmpose.pkl', 'wb') as file:
            pickle.dump(df_predict_mmpose, file)
            
        out.release()
        end = time.time()
        print('Prediction time:', end-start, 'secs')
        print('Done......')

        # Fermer la vidéo et les fenêtres OpenCV
        cap.release()
        cv2.destroyAllWindows()  

if __name__ == '__main__':
	# Création de l'objet ArgumentParser
	parser = argparse.ArgumentParser(description='Complete Tracking Script')

	# Ajout des arguments
	parser.add_argument('--inputs_path', type=str, help='Chemin vers le dossier d\'entrée')
	parser.add_argument('--outputs_path', type=str, help='Chemin vers le dossier de sortie')

	import torch
	print(torch.cuda.is_available())

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
	print(videos)
	print(len(videos))
	references = lister_fichiers(args.inputs_path,['.jpg','.png'])
	print(references)
	print(len(references))
	csv_court_coordinates = lister_fichiers(args.inputs_path,['.csv','.xlsx'])
	print(csv_court_coordinates)


	if len(videos)!=len(references):
		print('Something weird happened as there are '+str(len(videos))+' videos for '+str(len(references))+' references')
		sys.exit()
		
	if len(csv_court_coordinates)!=1:
		if len(csv_court_coordinates)==0:
			print('The court coordinates file has not been found at the path: '+args.inputs_path)
			sys.exit()
		else:
			print(len(csv_court_coordinates)+' csv/xlsx files have been found, there must be an issue')
			sys.exit()

    # Appel de la fonction complete_tracking avec les arguments fournis
	complete_tracking(args.inputs_path,videos,references,csv_court_coordinates, args.outputs_path)


