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
    global capture_count

    # Initialize capture count
    capture_count = 0

    # Initialize video stream
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            print('Error: Failed to capture image')
            break

        cv2.imshow('Pothole Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('c'):  # Capture image
            capture_count += 1
            if capture_count == 10:
                send_email()
                break

    video.release()
    cv2.destroyAllWindows()

def send_email():
    global from_location_entry, to_location_entry, logged_in_user

    sender_email = "vishnudevp2003@gmail.com"  # Your email address
    receiver_emails = ["vishnudevp@yahoo.com", "arunmgcomsci@gmail.com"]  # List of receiver email addresses
    password = "hqqi obps wuop azjg"  # Your email password

    from_location = from_location_entry.get()
    to_location = to_location_entry.get()

    # Create message container - the correct MIME type is multipart/alternative
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Emergency: Pothole Detected! Alert"
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)

    # Create the body of the message (a plain-text and an HTML version)
    text = f"Pothole has been detected between {from_location} and {to_location}."
    html = f"""\
    <html>
      <body>
        <p>Pothole has been detected between {from_location} and {to_location}. Please act immediately.</p>
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

    # Create secure connection with server and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_emails, msg.as_string()
        )

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

# Main application loop
app.mainloop()
