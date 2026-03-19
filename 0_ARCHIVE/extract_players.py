import pandas as pd
import numpy as np
import pickle
import cv2
import math as m


#Projection des 6 personnes détectées en 2D dans le monde réel, et extraction des joueurs du haut et du bas
def extract_players(videoId,inputs_path,outputs_path, court_coordinates):

    # Recharger le dataframe avec les tableaux d'origine depuis le fichier
    with open(outputs_path+'/'+videoId+'_df_predict.pkl', 'rb') as file:
        df_predict = pickle.load(file)

    # Paramètres du court sur l'image de référence    
    courts_pixel_coordinates = pd.read_excel(inputs_path+'/'+court_coordinates)

    court = courts_pixel_coordinates[courts_pixel_coordinates['videoId']==int(videoId)]
    top_left = list(map(float,court['top-left'].values[0].split(',')))
    top_right = list(map(float,court['top-right'].values[0].split(',')))
    bottom_left = list(map(float,court['bottom-left'].values[0].split(',')))
    bottom_right = list(map(float,court['bottom-right'].values[0].split(',')))
    # t_top = list(map(float,court['t-top'].values[0].split(',')))
    # t_bottom = list(map(float,court['t-bottom'].values[0].split(',')))
    court_coordinates = np.array([top_left,top_right,bottom_left,bottom_right], dtype=np.float32)

    width = court['width'].values[0]
    height = court['height'].values[0]

    #Equivalent dans le monde réel
    real_coordinates = np.array([[0, 0], [6.1, 0], [0, 13.4], [6.1, 13.4]], dtype=np.float32)

    # Calculate the perspective transformation matrix using the homography matrix
    M, _ = cv2.findHomography(court_coordinates, real_coordinates)

    def find_closest_player(players,previous_coords):
        distance_min = 10000
        index_min = 10
        for index, player in enumerate(players):
            distance = m.dist([player[0],player[1]],previous_coords)
            if distance<distance_min:
                distance_min = distance
                index_min = index
        return players[index_min]


    #Extraction des joueurs
    previous_coords_top = court_coordinates[0]
    previous_coords_bottom = court_coordinates[2]

    df_keypoints = pd.DataFrame(columns=['Frame','keypoints_top_player','keypoints_bottom_player'])
    df_keypoints = df_keypoints.set_index('Frame')

    df_court_coordinates = pd.DataFrame(columns=['Frame','court_coordinates'])
    df_court_coordinates = df_court_coordinates.set_index('Frame')

    #On parcourt toutes les frames
    for index, row in df_predict.iterrows():
        keypoints_with_scores = row['Keypoints_with_scores']

        player_top = []
        player_bottom = []

        df_court_coordinates.loc[index] = [court_coordinates]
        
        #On parcourt toutes les personnes (6)
        for person, keypoints in enumerate(keypoints_with_scores):

            shaped = np.squeeze(np.multiply(keypoints, [height,width,1]))

            left_ankle = shaped[15]
            right_ankle = shaped[16]

            left_ankle_y, left_ankle_x, left_ankle_conf = left_ankle
            right_ankle_y, right_ankle_x, right_ankle_conf = right_ankle

            #Seuil de confiance 
            if left_ankle_conf>0.1 or right_ankle_conf>0.1:

                # Define the pixel coordinates for the player
                player_pixel_coords = np.array([(left_ankle_x+right_ankle_x)/2, (left_ankle_y+right_ankle_y)/2], dtype=np.float32)

                # Apply the perspective transformation matrix to the player pixel coordinates
                player_real_coords = cv2.perspectiveTransform(player_pixel_coords.reshape(-1, 1, 2), M)

                coords_x = player_real_coords[0][0][0]
                coords_y = player_real_coords[0][0][1]

                #Dans la largeur du terrain
                if coords_x>-0.5 and coords_x<6.6:
                    
                    #Dans le terrain du haut
                    if coords_y>-1.5 and coords_y<6.7:

                        coords_x = 41 + coords_x*360/6.10
                        coords_y = 48 + coords_y*791/13.4 
                        
                        player_top.append([coords_x,coords_y,keypoints])

                    #Dans le terrain du bas
                    elif coords_y>6.7 and coords_y<13.9:

                        coords_x = 41 + coords_x*360/6.10
                        coords_y = 48 + coords_y*791/13.4 
                        
                        player_bottom.append([coords_x,coords_y,keypoints])
                    
                    # else:
                    #     coords_x = 41 + coords_x*360/6.10
                    #     coords_y = 48 + coords_y*791/13.4 
                        
                    #     other_players.append([coords_x,coords_y,keypoints])
                        
        
        #Aucun joueur trouvé
        # if len(player_top)==0 and len(other_players)>0:
        #     player_top = [find_closest_player(other_players,previous_coords_top)]
            
        # if len(player_bottom)==0 and len(other_players)>0:
        #     player_bottom = [find_closest_player(other_players,previous_coords_bottom)]
            
        #2 joueurs ou plus trouvés
        if len(player_top)>1:
            player_top = [find_closest_player(player_top,previous_coords_top)]
        elif len(player_bottom)>1:
            player_bottom = [find_closest_player(player_bottom,previous_coords_bottom)]
        
        #On enregistre les coordonnées du joueurs et on trace un cercle sur le court.
        if len(player_top)>0:
            previous_coords_top = [player_top[0][0],player_top[0][1]]
        if len(player_bottom)>0:
            previous_coords_bottom = [player_bottom[0][0],player_bottom[0][1]]
        
        if len(player_top)==0 and len(player_bottom)!=0:
            df_keypoints.loc[index] = [np.zeros((17,3)),player_bottom[0][2]]
        elif len(player_bottom)==0 and len(player_top)!=0:
            df_keypoints.loc[index] = [player_top[0][2],np.zeros((17,3))]
        elif len(player_top)==0 and len(player_bottom)==0:
            df_keypoints.loc[index] = [np.zeros((17,3)),np.zeros((17,3))]
        else:
            df_keypoints.loc[index] = [player_top[0][2],player_bottom[0][2]]

    df_predict = pd.concat([df_predict,df_keypoints],axis=1)
    df_predict = pd.concat([df_predict,df_court_coordinates],axis=1)
    # Sauvegarder le dataframe avec les tableaux d'origine à l'aide de Pickle
    with open(outputs_path+'/'+videoId+'_df_predict_with_players.pkl', 'wb') as file:
        pickle.dump(df_predict, file)

    return