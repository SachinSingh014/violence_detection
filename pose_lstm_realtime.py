import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load and modify model function
def load_and_modify_model(model_path):
    try:
        # Try to load the model normally
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
        return model
    except ValueError as e:
        print(f"Error loading model: {e}")
        print("Attempting to remove 'time_major' argument directly from the model file.")

        # Open the .h5 file and remove the 'time_major' argument
        with h5py.File(model_path, 'r+') as f:
            def visit_func(name, obj):
                if isinstance(obj, h5py.AttributeManager):
                    if 'time_major' in obj.attrs:
                        print(f"Removing 'time_major' from {name}")
                        del obj.attrs['time_major']

            f.visititems(visit_func)

        # Reload the model after modification
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully after modification.")
            return model
        except Exception as e:
            print(f"Failed to load the model after modification: {e}")
            return None

# Load model
model = load_and_modify_model(r"C:\Users\sachin\Desktop\violvipul\lstm-model.h5")

# Initialize variables
lm_list = []
label = "neutral"
neutral_label = "ntdtset"

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    if result[0][0] > 0.5:
        label = "violent" 
    else:
        label = "neutral"
    return str(label)

i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    i += 1

    if i > warm_up_frames:
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            if len(lm_list) == 20:
                # Use threading to avoid blocking the main loop
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []
            
            # Draw bounding box around the landmarks
            x_coordinate = []
            y_coordinate = []
            for lm in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_coordinate.append(cx)
                y_coordinate.append(cy)
            cv2.rectangle(frame,
                            (min(x_coordinate), max(y_coordinate)),
                            (max(x_coordinate), min(y_coordinate) - 25),
                            (0, 255, 0),
                            1)

            frame = draw_landmark_on_image(mpDraw, results, frame)

        frame = draw_class_on_image(label, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")

# Release resources
cap.release()
cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import threading

# # Initialize video capture
# cap = cv2.VideoCapture(0)

# # Initialize MediaPipe Pose
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# # Define custom objects for model loading
# custom_objects = {
#     'Orthogonal': tf.keras.initializers.Orthogonal
# }

# # Load model
# def load_model(model_path):
#     return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# model = load_model(r"C:\Users\sachin\Desktop\violvipul\lstm-model.h5")

# # Initialize variables
# lm_list = []
# label = "neutral"
# neutral_label = "ntdtset"

# def make_landmark_timestep(results):
#     c_lm = []
#     for lm in results.pose_landmarks.landmark:
#         c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
#     return c_lm

# def draw_landmark_on_image(mpDraw, results, frame):
#     mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#     for lm in results.pose_landmarks.landmark:
#         h, w, _ = frame.shape
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#     return frame

# def draw_class_on_image(label, img):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     bottomLeftCornerOfText = (10, 30)
#     fontScale = 1
#     fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
#     thickness = 2
#     lineType = 2
#     cv2.putText(img, str(label),
#                 bottomLeftCornerOfText,
#                 font,
#                 fontScale,
#                 fontColor,
#                 thickness,
#                 lineType)
#     return img

# def detect(model, lm_list):
#     global label
#     lm_list = np.array(lm_list)
#     lm_list = np.expand_dims(lm_list, axis=0)
#     result = model.predict(lm_list)
#     if result[0][0] > 0.5:
#         label = "violent" 
#     else:
#         label = "neutral"
#     return str(label)

# i = 0
# warm_up_frames = 60

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frameRGB)
#     i += 1

#     if i > warm_up_frames:
#         if results.pose_landmarks:
#             lm = make_landmark_timestep(results)
#             lm_list.append(lm)

#             if len(lm_list) == 20:
#                 # Use threading to avoid blocking the main loop
#                 t1 = threading.Thread(target=detect, args=(model, lm_list))
#                 t1.start()
#                 lm_list = []
            
#             # Draw bounding box around the landmarks
#             x_coordinate = []
#             y_coordinate = []
#             for lm in results.pose_landmarks.landmark:
#                 h, w, _ = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 x_coordinate.append(cx)
#                 y_coordinate.append(cy)
#             cv2.rectangle(frame,
#                             (min(x_coordinate), max(y_coordinate)),
#                             (max(x_coordinate), min(y_coordinate) - 25),
#                             (0, 255, 0),
#                             1)

#             frame = draw_landmark_on_image(mpDraw, results, frame)

#         frame = draw_class_on_image(label, frame)
#         cv2.imshow("image", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break

# df = pd.DataFrame(lm_list)
# df.to_csv(label + ".txt")

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
