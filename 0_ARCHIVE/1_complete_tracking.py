from TrackNetv2.three_in_three_out.predict3 import *
from multipose_movenet import *
#from run_mmpose import *
from camera_changes import *
import pandas as pd
import pickle

import argparse
import os
import sys

# import mmcv
# from mmcv import imread
# import mmengine
# from mmengine.registry import init_default_scope

# from mmpose.apis import inference_topdown
# from mmpose.apis import init_model as init_pose_estimator
# from mmpose.evaluation.functional import nms
# from mmpose.registry import VISUALIZERS
# from mmpose.structures import merge_data_samples

# try:
#     from mmdet.apis import inference_detector, init_detector
#     has_mmdet = True
# except (ImportError, ModuleNotFoundError):
#     has_mmdet = False
    

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
		
		#On charge le model Tracknet
		load_weights = "./TrackNetv2/three_in_three_out/model906_30"
		model_tracknet = load_model(load_weights, custom_objects={'custom_loss':custom_loss})
		#model_tracknet.summary()

		#On charge le model movenet
		model_movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
		movenet = model_movenet.signatures['serving_default']

		#Détection des scènes avec angle classique
		print("Beginning detecting similar scenes")
		similar_scenes = detect_similar_camera(video_path,reference_frame_path)
		print("Done...")

		#On ouvre la vidéo: 
		cap = cv2.VideoCapture(video_path)
		ret, image1 = cap.read()
		ratio = image1.shape[0] / HEIGHT
		size = (int(WIDTH*ratio), int(HEIGHT*ratio))
		fps = cap.get(cv2.CAP_PROP_FPS)

		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(outputs_path+'/'+videoName[:-4]+'_predict.mp4', fourcc, fps, size)

		print('Beginning predicting......')

		start = time.time()

		#On définit un dataframe que l'on va sauvegarder
		df_predict_tracknet = pd.DataFrame(columns=['Frame', 'Visibility_Shuttle','ShuttleX','ShuttleY','Time'])
		df_predict_tracknet = df_predict_tracknet.set_index('Frame')
		df_predict_movenet = pd.DataFrame(columns=['Frame','Keypoints_with_scores', 'Boxes_with_scores'])

		#On parcourt les scènes qui nous intéressent
		for scene in similar_scenes:
			start_frame = scene[0].get_frames()
			end_frame = scene[1].get_frames()

			# Set the video frame position to the start of the scene
			cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

			# Process frames within the scene
			for frame_number in range(start_frame, end_frame + 1,3):
				success, image1 = cap.read()
				frame_time1 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
				success, image2 = cap.read()
				frame_time2 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
				success, image3 = cap.read()
				frame_time3 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))

				### Tracking du volant ###

				unit = []
				#Adjust BGR format (cv2) to RGB format (PIL)
				x1 = image1[...,::-1]
				x2 = image2[...,::-1]
				x3 = image3[...,::-1]
				#Convert np arrays to PIL images
				x1 = array_to_img(x1)
				x2 = array_to_img(x2)
				x3 = array_to_img(x3)
				#Resize the images
				x1 = x1.resize(size = (WIDTH, HEIGHT))
				x2 = x2.resize(size = (WIDTH, HEIGHT))
				x3 = x3.resize(size = (WIDTH, HEIGHT))
				#Convert images to np arrays and adjust to channels first
				x1 = np.moveaxis(img_to_array(x1), -1, 0)
				x2 = np.moveaxis(img_to_array(x2), -1, 0)
				x3 = np.moveaxis(img_to_array(x3), -1, 0)
				#Create data
				unit.append(x1[0])
				unit.append(x1[1])
				unit.append(x1[2])
				unit.append(x2[0])
				unit.append(x2[1])
				unit.append(x2[2])
				unit.append(x3[0])
				unit.append(x3[1])
				unit.append(x3[2])
				unit=np.asarray(unit)	
				unit = unit.reshape((1, 9, HEIGHT, WIDTH))
				unit = unit.astype('float32')
				unit /= 255
				y_pred = model_tracknet.predict(unit, batch_size=BATCH_SIZE)
				y_pred = y_pred > 0.5
				y_pred = y_pred.astype('float32')
				h_pred = y_pred[0]*255
				h_pred = h_pred.astype('uint8')
				for i in range(3):
					if i == 0:
						frame_time = frame_time1
						image = image1
					elif i == 1:
						frame_time = frame_time2
						image = image2
					elif i == 2:
						frame_time = frame_time3	
						image = image3

					if np.amax(h_pred[i]) <= 0:
						df_predict_tracknet.loc[frame_number+i] = [0,0,0,frame_time]
					else:	
						#h_pred
						(cnts, _) = cv2.findContours(h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
						rects = [cv2.boundingRect(ctr) for ctr in cnts]
						max_area_idx = 0
						max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
						for j in range(len(rects)):
							area = rects[j][2] * rects[j][3]
							if area > max_area:
								max_area_idx = j
								max_area = area
						target = rects[max_area_idx]
						(cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))

						df_predict_tracknet.loc[frame_number+i] = [1,cx_pred,cy_pred,frame_time]
						#image_cp = np.copy(image)
						cv2.circle(image, (cx_pred, cy_pred), 5, (0,0,255), -1)

					### Estimation des positions ###
					# Resize image
					img = image.copy()
					img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 352,640)
					input_img = tf.cast(img, dtype=tf.int32)
					
					# Detection section
					results = movenet(input_img)
					keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
					boxes_with_scores = results['output_0'].numpy()[:,:,51:].reshape((6,1,5))
					
					# Render keypoints 
					loop_through_people(image, keypoints_with_scores, boxes_with_scores, EDGES, 0.15)
					
					# Ajouter les matrices dans le dataframe
					# df_predict_movenet.loc[frame_number+i, 'Keypoints_with_scores'] = keypoints_with_scores
					# df_predict_movenet.loc[frame_number+i, 'Boxes_with_scores'] = boxes_with_scores
					df_predict_movenet.loc[len(df_predict_movenet)] = [frame_number+i, keypoints_with_scores, boxes_with_scores]

					out.write(image)


		df_predict_movenet = df_predict_movenet.set_index('Frame')

		df_predict_tracknet = df_predict_tracknet[~df_predict_tracknet.index.duplicated(keep='first')]
		df_predict_movenet = df_predict_movenet[~df_predict_movenet.index.duplicated(keep='first')]

		df_predict = pd.concat([df_predict_tracknet,df_predict_movenet],axis=1)
		# Sauvegarder le dataframe avec les tableaux d'origine à l'aide de Pickle
		with open(outputs_path+'/'+videoName[:-4]+'_df_predict.pkl', 'wb') as file:
			pickle.dump(df_predict, file)
			
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
	print(args.outputs_path)
	complete_tracking(args.inputs_path,videos,references,csv_court_coordinates, args.outputs_path)
	


