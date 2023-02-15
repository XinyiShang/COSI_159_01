from deepface import DeepFace

class deepface:
    detection_model = 'Facenet'
    img_path1 = 'Task3\img1.jpg'
    img_path2 = 'Task3\img2.jpg'
    result = DeepFace.verify(img1_path = img_path1, img2_path = img_path2, model_name=detection_model)
    print(result)