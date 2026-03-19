import cv2
from scenedetect import detect, ContentDetector


#Fonction qui détecte les changements de scènes et va comparer les images avec une image de référence.
def detect_similar_camera(video_path,reference_frame_path):

    # Load the reference frame
    reference_frame = cv2.imread(reference_frame_path, cv2.IMREAD_GRAYSCALE)

    if reference_frame is None:
        print("Error: Unable to load the reference frame.")
        return  # Exit the function or handle the error as needed
    
    reference_frame = cv2.GaussianBlur(reference_frame, (21, 21), 0)

    # Open the video using OpenCV
    video = cv2.VideoCapture(video_path)

    # Check if the video capture is successful
    if not video.isOpened():
        print("Error opening video file:", video_path)
        exit()

    # Detect scene changes
    scene_list = detect(video_path, ContentDetector())

    # Iterate over each scene
    similar_scenes = []
    for scene in scene_list:
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        if end_frame - start_frame > 75:

            # Set the video frame position to the start of the scene
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Calculate the similarity between the reference frame and the entire scene
            scene_similarity = 0
            num_frames = 0

            # Process frames within the scene
            for frame_number in range(start_frame, end_frame + 1):
                ret, frame = video.read()
                if not ret:
                    break

                # Convert the frame to grayscale and apply Gaussian blur
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

                # Compare the frame with the reference frame
                frame_diff = cv2.absdiff(reference_frame, gray_frame)
                _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

                # Calculate the number of similar pixels
                num_similar_pixels = threshold.size - cv2.countNonZero(threshold)

                # Accumulate the similarity for each frame
                scene_similarity += num_similar_pixels
                num_frames += 1

            # Calculate the average similarity for the scene
            if num_frames > 0:
                avg_similarity = scene_similarity / (threshold.size * num_frames)

                # If the average similarity is above a certain threshold, consider it a similar scene
                if avg_similarity > 0.80:  # Adjust the threshold as needed
                    similar_scenes.append(scene)
                    print('Similar Scene found',scene)

    # Release the video and clean up
    video.release()
    #cv2.destroyAllWindows()

    return similar_scenes