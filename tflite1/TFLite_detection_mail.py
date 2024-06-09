from email.mime.image import MIMEImage
import customtkinter as ctk
import tkinter.messagebox as tkmb
import shelve
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Selecting GUI theme - dark, light , system (for system default)
ctk.set_appearance_mode("dark")

# Selecting color theme - blue, green, dark-blue
ctk.set_default_color_theme("blue")

user_entry = None
user_pass = None
logged_in_user = None
from_location_entry = None
to_location_entry = None
ready_button = None
capture_count = 0

def login():
    global logged_in_user
    username = user_entry.get()
    password = user_pass.get()

    with shelve.open("user_database.db") as db:
        if username in db:
            if db[username] == password:
                logged_in_user = username
                tkmb.showinfo(title="Login Successful", message=f"Welcome, {username}! You have logged in Successfully")
                clear_existing_widgets()
                create_welcome_page()
                return
            else:
                tkmb.showwarning(title='Login Failed', message='Incorrect Password. Please try again.')
        else:
            tkmb.showwarning(title='Login Failed', message='Username not found. Please sign up first.')

def signup():
    global user_entry, user_pass  # Declare global variables
    username = user_entry.get()
    password = user_pass.get()

    with shelve.open("user_database.db") as db:
        if username in db:
            tkmb.showwarning(title='Sign Up Failed', message='Username already taken. Please choose another username.')
        else:
            db[username] = password
            tkmb.showinfo(title="Sign Up Successful", message="You have signed up successfully. Please login.")

def create_signup_page():
    global user_entry, user_pass  # Declare global variables

    clear_existing_widgets()

    label = ctk.CTkLabel(app, text='SIGNUP PAGE')
    label.pack(pady=12, padx=10)

    # Username
    user_entry = ctk.CTkEntry(app, placeholder_text="Username")
    user_entry.pack(pady=12, padx=10)

    # Password
    user_pass = ctk.CTkEntry(app, placeholder_text="Password", show="*")
    user_pass.pack(pady=12, padx=10)

    # Sign Up Button
    button_signup = ctk.CTkButton(app, text='Sign Up', command=signup)
    button_signup.pack(pady=12, padx=10)

    # Back Button
    button_back = ctk.CTkButton(app, text='Back', command=create_login_page)
    button_back.pack(pady=12, padx=10)


def create_login_page():
    global user_entry, user_pass  # Declare global variables

    clear_existing_widgets()

    label = ctk.CTkLabel(app, text="")
    label.pack(pady=20)

    frame = ctk.CTkFrame(app)
    frame.pack(pady=20, padx=40, fill='both', expand=True)

    label = ctk.CTkLabel(frame, text='LOGIN PAGE')
    label.pack(pady=12, padx=10)

    user_entry = ctk.CTkEntry(frame, placeholder_text="Username")
    user_entry.pack(pady=12, padx=10)

    user_pass = ctk.CTkEntry(frame, placeholder_text="Password", show="*")
    user_pass.pack(pady=12, padx=10)

    button_login = ctk.CTkButton(frame, text='Login', command=login)
    button_login.pack(pady=12, padx=5, anchor="center")

    button_signup = ctk.CTkButton(frame, text='Sign Up', command=create_signup_page)
    button_signup.pack(pady=12, padx=5, anchor="center")

    checkbox = ctk.CTkCheckBox(frame, text='Remember Me')
    checkbox.pack(pady=12, padx=10)

def create_welcome_page():
    global logged_in_user, from_location_entry, to_location_entry, ready_button

    clear_existing_widgets()

    label_user = ctk.CTkLabel(app, text=f"Welcome, {logged_in_user}!", font=("Algerian", 38, "bold"), anchor="center")
    label_user.pack(pady=20)

    label_from = ctk.CTkLabel(app, text="From:")
    label_from.pack(pady=5)
    from_location_entry = ctk.CTkEntry(app, placeholder_text="Enter from location")
    from_location_entry.pack(pady=5)

    label_to = ctk.CTkLabel(app, text="To:")
    label_to.pack(pady=5)
    to_location_entry = ctk.CTkEntry(app, placeholder_text="Enter to location")
    to_location_entry.pack(pady=5)

    ready_button = ctk.CTkButton(app, text='READY FOR POTHOLE DETECTING', command=check_location_and_start_detection)
    ready_button.pack(pady=20)

    button_quit = ctk.CTkButton(app, text='Quit', command=app.quit)
    button_quit.pack(pady=10)

def check_location_and_start_detection():
    global from_location_entry, to_location_entry
    from_location = from_location_entry.get()
    to_location = to_location_entry.get()

    if from_location.strip() == "" or to_location.strip() == "":
        tkmb.showwarning(title='Incomplete Information', message='Please fill in both from and to locations.')
    else:
        clear_existing_widgets()
        start_detection()


def start_detection():
    global frame_rate_calc, capture_count

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Classindex of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Check if pothole is detected
                if object_name == "Pothole":
                    # Capture the image when pothole is detected
                    image_path = f"Pothole_{capture_count}.jpg"
                    cv2.imwrite(image_path, frame)  # Saving the frame as an image
                    capture_count += 1
                    # Send the captured image as an email attachment
                    send_email(image_path)

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # Display frame
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()



def send_email(image_path):
    global capture_count  # Assuming capture_count is a global variable

    sender_email = "vishnudevp2003@gmail.com"  # Your email address
    receiver_emails = ["vishnudevp@yahoo.com", "arunmgcomsci@gmail.com"]  # List of receiver email addresses
    password = "hqqi obps wuop azjg"  # Your email password

    # Create message container - the correct MIME type is multipart/related
    msg = MIMEMultipart('related')
    msg['Subject'] = "Emergency: Pothole Detected! Alert"
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)

    # Create the body of the message (a plain-text and an HTML version)
    text = f"Pothole has been detected."
    html = f"""\
    <html>
      <body>
        <p>Pothole has been detected. Please act immediately.</p>
        <img src="cid:image1">
      </body>
    </html>
    """

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    msg.attach(part1)
    msg.attach(part2)

    # Add attachment image
    with open(image_path, 'rb') as attachment:
        image_part = MIMEImage(attachment.read(), name=f"Pothole_{capture_count}.jpg")
        image_part.add_header('Content-ID', '<image1>')
        msg.attach(image_part)

    try:
        # Create secure connection with server and send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_emails, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")




def clear_existing_widgets():
    # Destroy all widgets currently present in the app
    for widget in app.winfo_children():
        widget.destroy()

# Main application
app = ctk.CTk()
app.geometry("400x400")
app.title("Modern Login UI using Customtkinter")

# Initial creation of login page
create_login_page()

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Main application loop
app.mainloop()
