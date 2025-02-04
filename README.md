# Intelligent-Pothole-Detection-System

![Screenshot (10)](https://github.com/user-attachments/assets/52396e9a-8d91-46f8-9e67-ef82a3cba89e)


Utilizes computer vision and machine learning for real-time pothole detection via webcams. Provides efficient and cost-effective monitoring, aiding timely road repairs through cloud-based reporting for government agencies
How to Run TensorFlow Lite Models on Windows

This guide shows how to set up a TensorFlow Lite Runtime environment on a Windows PC. We'll use Anaconda to create a Python environment to install the TFLite Runtime in. It's easy!

Step 1. Download and Install Anaconda
First, install Anaconda, which is a Python environment manager that greatly simplifies Python package management and deployment. Anaconda allows you to create Python virtual environments on your PC without interfering with existing installations of Python. Go to the Anaconda Downloads page and click the Download button.
When the download finishes, open the downloaded .exe file and step through the installation wizard. Use the default install options.


Step 2. Set Up Virtual Environment and Directory
Go to the Start Menu, search for "Anaconda Command Prompt", and click it to open up a command terminal. We'll create a folder called tflite1 directly in the C: drive. (You can use any other folder location you like, just make sure to modify the commands below to use the correct file paths.) Create the folder and move into it by issuing the following commands in the terminal:
mkdir C:\tflite1
cd C:\tflite1

Next, create a Python 3.9 virtual environment by issuing:
conda create --name tflite1-env python=3.9

Enter "y" when it asks if you want to proceed. Activate the environment and install the required packages by issuing the commands below. We'll install TensorFlow, OpenCV, and a downgraded version of protobuf. TensorFlow is a pretty big download (about 450MB), so it will take a while.
conda activate tflite1-env
pip install tensorflow opencv-python protobuf==3.20.*


Download the detection scripts from this repository by issuing:
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_image.py --output TFLite_detection_image.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_video.py --output TFLite_detection_video.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_webcam.py --output TFLite_detection_webcam.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_stream.py --output TFLite_detection_stream.py


Step 3. Move TFLite Model into Directory
Next, take the custom TFLite model that was trained and downloaded from the Colab notebook and move it into the C:\tflite1 directory. If you downloaded it from Colab, it should be in a file called custom_model_lite.zip. (If you haven't trained a model yet and just want to test one out, download my "bird, squirrel, raccoon" model by clicking this Dropbox link.) Move that file to the C:\tflite1 directory. Once it's moved, unzip it using:
tar -xf custom_model_lite.zip

At this point, you should have a folder at C:\tflite1\custom_model_lite which contains at least a detect.tflite and labelmap.txt file.

Step 4. Run TensorFlow Lite Model!
Alright! Now that everything is set up, running the TFLite model is easy. Just call one of the detection scripts and point it at your model folder with the --modeldir option. For example, to run your custom_model_lite model on a webcam, issue:
python TFLite_detection_webcam.py --modeldir=custom_model_lite

A window will appear showing detection results drawn on the live webcam feed. For more information on how to use the detection scripts, please see Step 3 in the main README page.

Have fun using TensorFlow Lite! Stay tuned for more examples on how to build cool applications around your model.




NOTE:if you have cloned the repositiory then it is added to folder called tflite1 then just do from the step3 and no need of step1 and step2.
