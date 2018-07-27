###########configurations, modify any of them if needed
from AllConfigs import Config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' ### we set this to a idle GPU so we can monitor the GPU memory usage, please change to 0 if you have only one GPU
loop_c = 1000
wait_p = 70

### create the class for loading emotion recognition
import torch.nn as nn
class Dupe_Net(nn.Module):
    def __init__(self):
        super(Dupe_Net, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 15, 5),
            # nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(15),
            nn.Conv2d(15, 30, 5),
            # nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 60, 5),
            # nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 60, 3),
            # nn.Dropout(0.2),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(60 * 2 * 2, 160),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(160, 80),
            nn.Dropout(0.5),
        )
        self.smx1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(80, 7),
            nn.Dropout(0.5),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.cnn1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.smx1(x)
        return x

### load pre-trained model of emotion recognition, 6emo classification is applied#############
import torch
emotions = ["anger", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]  # Emotion list
net = Dupe_Net()
if torch.cuda.is_available():
    net.cuda()
    net = nn.DataParallel(net) ## a fake parallel gpu load since the model is trained in Multi-gpu context
net.load_state_dict(torch.load(Config.hao_emotion_recognition_model_path))
net.eval()
print("Progress Report: Emotion recognition model loading complete~")


### import ibug_face_tracker, not in the same project so load path in config file
import sys
sys.path.append(Config.ibug_face_tracker_path)
import ibug_face_tracker
from ibug_face_tracker import FaceTracker ### ignore possible compiler warning of this line
#### load ibug_face_tracker ######
tracker = FaceTracker(os.path.realpath(Config.ert_model_path),
                      os.path.realpath(Config.auxiliary_model_path))
tracker.hard_failure_threshold = float(Config.hard_failure_threshold)
tracker.soft_failure_threshold = float(Config.soft_failure_threshold)
tracker.maximum_number_of_soft_failures = int(Config.maximum_number_of_soft_failures)
tracker.failure_detection_interval = 1
tracker.minimum_face_detection_gap = 2
tracker.minimum_face_size = 64
print('Landmark tracker initialised.')

#### define basic oprations for extract face from image ########
#### these functions could also be found in smile_detection_pipeline/misc.py########
import cv2
import math
import numpy as np
def get_face_box(landmarks, margin=(0.0, 0.0, 0.0, 0.0), exclude_chin_points=False):
    # Get bounding box
    if exclude_chin_points and landmarks.shape[0] == 68:
        top_left = np.min(landmarks[17:], axis=0)
        bottom_right = np.max(landmarks[17:], axis=0)
    else:
        top_left = np.min(landmarks, axis=0)
        bottom_right = np.max(landmarks, axis=0)
    face_size = bottom_right - top_left
    top_left[0] = np.floor(top_left[0] - face_size[0] * margin[0])
    top_left[1] = np.floor(top_left[1] - face_size[1] * margin[1])
    bottom_right[0] = np.ceil(bottom_right[0] + face_size[0] * margin[2])
    bottom_right[1] = np.ceil(bottom_right[1] + face_size[1] * margin[3])

    # Make face box square
    difference = (bottom_right[1] - top_left[1] + 1) - (bottom_right[0] - top_left[0] + 1)
    if difference > 0:
        top_left[0] -= difference // 2
        bottom_right[0] += difference - difference // 2
    elif difference < 0:
        difference = -difference
        top_left[1] -= difference // 2
        bottom_right[1] += difference - difference // 2

    return top_left, bottom_right
def extract_face_image(image, landmarks, target_size, margin, head_pose=None,
                       eye_points=None, exclude_chin_points=False,
                       interpolation=cv2.INTER_CUBIC):
    # First, see whether we have head pose
    if head_pose is not None and len(head_pose) >= 3:
        roll = head_pose[2]
    else:
        roll = None

    # Get a proper bounding box for the face
    if margin is None:
        margin = (0.0, 0.0, 0.0, 0.0)
    if roll is None:
        top_left, bottom_right = get_face_box(landmarks, margin, exclude_chin_points)
    else:
        rotated_landmarks = cv2.getRotationMatrix2D((0, 0), roll, 1.0).dot(
            np.vstack((landmarks.T, np.ones(landmarks.shape[0])))).T
        top_left, bottom_right = get_face_box(rotated_landmarks, margin, exclude_chin_points)
        corners = np.array([[top_left[0], top_left[1], 1],
                            [bottom_right[0], top_left[1], 1],
                            [bottom_right[0], bottom_right[1], 1],
                            [top_left[0], bottom_right[1], 1]])
        top_left, bottom_right = get_face_box(cv2.getRotationMatrix2D(
            (0, 0), -roll, 1.0).dot(corners.T).T, (0, 0, 0, 0), exclude_chin_points)
        top_left -= 5
        bottom_right += 5

    # Enlarge the image if necessary
    padding = np.zeros((image.ndim, 2), dtype=int)
    if top_left[0] < 0:
        padding[1][0] = -int(top_left[0])
    if top_left[1] < 0:
        padding[0][0] = -int(top_left[1])
    if bottom_right[0] >= image.shape[1]:
        padding[1][1] = int(bottom_right[0] - image.shape[1] + 1)
    if bottom_right[1] >= image.shape[0]:
        padding[0][1] = int(bottom_right[1] - image.shape[0] + 1)
    image = np.pad(image, padding, 'symmetric')

    # Revise bounding box and landmarks accordingly
    landmarks = landmarks.copy()
    if eye_points is not None:
        eye_points = eye_points.copy()
    if top_left[0] < 0:
        bottom_right[0] -= top_left[0]
        landmarks[:, 0] -= top_left[0]
        if eye_points is not None:
            eye_points[:, 0] -= top_left[0]
        top_left[0] = 0
    if top_left[1] < 0:
        bottom_right[1] -= top_left[1]
        landmarks[:, 1] -= top_left[1]
        if eye_points is not None:
            eye_points[:, 1] -= top_left[1]
        top_left[1] = 0

    # Extract the face image
    face_image = image[int(top_left[1]): int(bottom_right[1] + 1),
                       int(top_left[0]): int(bottom_right[0] + 1)]
    landmarks[:, 0] -= top_left[0]
    landmarks[:, 1] -= top_left[1]
    if eye_points is not None:
        eye_points[:, 0] -= top_left[0]
        eye_points[:, 1] -= top_left[1]

    # If head pose is given, rotate everything
    if roll is not None:
        rotation_matrix = cv2.getRotationMatrix2D(((face_image.shape[1] - 1) / 2.0,
                                                   (face_image.shape[0] - 1) / 2.0), roll, 1.0)
        face_image = cv2.warpAffine(face_image, rotation_matrix,
                                    dsize=(face_image.shape[1], face_image.shape[0]),
                                    flags=(interpolation + cv2.WARP_FILL_OUTLIERS))
        landmarks = rotation_matrix.dot(np.vstack((landmarks.T, np.ones(landmarks.shape[0])))).T
        if eye_points is not None:
            eye_points = rotation_matrix.dot(np.vstack((eye_points.T, np.ones(eye_points.shape[0])))).T
        top_left, bottom_right = get_face_box(landmarks, margin, exclude_chin_points)
        if top_left[0] > 0:
            landmarks[:, 0] -= top_left[0]
            eye_points[:, 0] -= top_left[0]
        if top_left[1] > 0:
            landmarks[:, 1] -= top_left[1]
            eye_points[:, 1] -= top_left[1]
        face_image = face_image[max(int(top_left[1]), 0): min(int(bottom_right[1] + 1), face_image.shape[0]),
                                max(int(top_left[0]), 0): min(int(bottom_right[0] + 1), face_image.shape[1])]

    # Rescale landmarks and face image
    if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
        landmarks[:, 0] *= float(target_size[0]) / max(face_image.shape[1], 1)
        landmarks[:, 1] *= float(target_size[1]) / max(face_image.shape[0], 1)
        if eye_points is not None:
            eye_points[:, 0] *= float(target_size[0]) / max(face_image.shape[1], 1)
            eye_points[:, 1] *= float(target_size[1]) / max(face_image.shape[0], 1)
        face_image = cv2.resize(face_image, target_size, interpolation=interpolation)
    else:
        face_image = face_image.copy()

    return face_image, landmarks, eye_points
def gamma_correction(image, target_mean):
    target_mean = target_mean / 255.0
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        image_mean = image[:, :, 2].mean() / 255.0
        gamma = math.log(target_mean, image_mean)
        image[:, :, 2] = np.round(np.power(image[:, :, 2] / 255.0, gamma) * 255.0).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR_FULL)
    else:
        image_mean = image.mean() / 255.0
        gamma = math.log(target_mean, image_mean)
        return np.round(np.power(image / 255.0, gamma) * 255.0).astype(np.uint8)

#### define the main function to run #############
def main():
    ### open the camera ####
    webcam = cv2.VideoCapture(0)
    if webcam.isOpened():
        print('Webcam #0 opened.')
    ####start the loop to get frame from camera########
    for x in range(loop_c):
        _, frame = webcam.read()
        tracker.track(frame)
        #############start only if face is found##############
        if tracker.has_facial_landmarks:
            face_image, aligned_landmarks, aligned_eye_points = extract_face_image(
                frame, tracker.facial_landmarks, (224,224),
                (0.05, 0.1, 0.05, 0.1), (tracker.pitch, tracker.yaw, tracker.roll),
                tracker.eye_landmarks)
            if Config.use_gamma_correction:
                face_image = gamma_correction(face_image, 88.69477439413265)
            face_image2 = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            ########prepare face with landmark image to show#########
            ibug_face_tracker.FaceTracker.plot_landmark_connections(face_image, aligned_landmarks)
            ibug_face_tracker.FaceTracker.plot_facial_landmarks(face_image, aligned_landmarks)
            if aligned_eye_points is not None:
                ibug_face_tracker.FaceTracker.plot_eye_landmarks(face_image, aligned_eye_points)
            face_image = cv2.flip(face_image, 1)
            cv2.namedWindow("Face with Landmarks", 0);
            cv2.resizeWindow("Face with Landmarks", 500, 500);
            cv2.moveWindow("Face with Landmarks", 70, 120);
            cv2.imshow('Face with Landmarks', face_image)

            ########prepare data to fit in model. input faces must fit the correct boarder of training image##########
            image2 = cv2.resize(face_image2, (350, 350))
            shifted = image2[21:310, 31:320]
            image = cv2.resize(shifted, (64, 64))  # Resize face so all images have same size
            crop_img = image[2:62, 2:62]
            imageT = np.zeros((1, 1, 60, 60), dtype=np.float)
            # normalized the image, the value is same as in training
            imageT[0, 0] = crop_img / 255.0
            imageT = (imageT - 0.485) / 0.229
            ############ make predicition ###############
            imageT = torch.from_numpy(imageT).float()
            if torch.cuda.is_available():
                imageT = imageT.cuda()
            pred = net(imageT)
            print pred.data
            _, predicted = torch.max(pred.data, 1)
            result = emotions[predicted]

            ################ prepare face with emotion prediction to show ##################
            cv2.namedWindow("Detected", 0);
            cv2.resizeWindow("Detected", 500, 500);
            image2 = cv2.flip(image2,1)
            cv2.putText(image2, result, (20, 330),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(255, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.moveWindow("Detected", 670, 120);

            cv2.imshow("Detected", image2)
            cv2.waitKey(wait_p)

    print "main process ends"


if __name__ == "__main__":
    main()
