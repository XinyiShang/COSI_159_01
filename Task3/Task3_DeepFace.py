import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
"""
def load_images(image_dir,Num):
    # Load a set of images from a directory
    image_files = [os.path.join(image_dir, f"img{i}.jpg") for i in range(1, Num+1)]
    images = []
    for image_file in image_files:
        img = Image.open(image_file)
        images.append(img)
    return images
"""
"""
def clustering():
    # Get the list of images to cluster
    image_dir = "Task3"
    image_files = load_images (image_dir,14)

    # Compute the Deepface feature vectors for each image
    feature_vectors = []
    for image_file in image_files:
        features = DeepFace.represent(image_file, model_name="Facenet")
        feature_vectors.append(features)

    # Convert the feature vectors to a numpy array
    X = np.array(feature_vectors)

    # Apply the k-means algorithm to cluster the images
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    # Print the cluster assignments for each image
    for i, image_file in enumerate(image_files):
        cluster_id = kmeans.labels_[i]
        print(f"{image_file} belongs to cluster {cluster_id}")
"""

def main():
    Task = "detection"
    if Task == "verification":
    #try some verifications 
        detection_model = 'Facenet'
        task_number = 4
        if task_number == 1:
            img_path1 = 'Task3\img1.jpg'
            img_path2 = 'Task3\img5.jpg'
        elif task_number ==2 :
            img_path1 = 'Task3\img7.jpg'
            img_path2 = 'Task3\img6.jpg'
        elif task_number ==3 :
            img_path1 = 'Task3\img9.jpg'
            img_path2 = 'Task3\img10.jpg'
        elif task_number ==4 :
            img_path1 = 'Task3\img12.jpg'
            img_path2 = 'Task3\img4.jpg'
        else:
            print("wrong task number")
        result = DeepFace.verify(img1_path = img_path1, img2_path = img_path2, model_name=detection_model)
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        plt.imshow(img1[:,:,::-1])
        plt.show()
        plt.imshow(img2[:,:,::-1])
        plt.show()

        print(result)

    elif Task == "attribute":
    #Try attribute analysis
        task_number = 3
        if task_number == 1:
            img_path = "Task3\img9.jpg"
        elif task_number == 2:
            img_path = "Task3\img10.jpg"
        elif task_number ==3:
            img_path = "Task3\img14.jpg"
        elif task_number ==4:
            img_path = "Task3\img2.jpg"
        objs = DeepFace.analyze(img_path = img_path, actions = ['age', 'gender', 'race', 'emotion'])
        print(objs)

    elif Task == "detection":
        backends = [
            'opencv', 
            'ssd', 
            'dlib', 
            'mtcnn', 
            'retinaface', 
            'mediapipe'
        ]

        # Load the image and extract faces
        img_path = "Task3\img1.jpg"

        # Extract faces from the input image
        faces = DeepFace.extract_faces(img_path)

        print(faces)
        plt.imshow(faces[0]['face'])
        plt.show()

    else:
        print("wrong task")

if __name__ == "__main__":
    main()


