import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import CNNModel
from skimage import feature
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import CNNModel
from keras import optimizers
import sqlite3
from tensorflow.keras.optimizers import SGD
global fn
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Driver face recognition")

image2 =Image.open('img67.jpg')
image2 =image2.resize((w,h), Image.LANCZOS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)
#
label=tk.Label(root,text="Driver face recognition",font=("Calibri",45),
               bg="black",fg="white",
               width=50,
               height=1)
label.place(x=0,y=0)  




frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=350, bd=5, font=('times', 14, ' bold '), fg='white',bg="black")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=120)



def update_label1(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='white', fg='black')
    result_label.place(x=300, y=550)
    
    
    
################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def update_cal(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='white', fg='black')
    result_label.place(x=350, y=400)
    
    
    
###########################################################################
def train_model():
 
    update_label("Model Training Start...............")
    
    start = time.time()

    X= CNNModel.main()
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    print(msg)

import functools
import operator


def convert_str_to_tuple(tup):
    s = functools.reduce(operator.add, (tup))
    return s

def test_model_proc(fn):
    from keras.models import load_model
    from tensorflow.keras.optimizers import Adam

    IMAGE_SIZE = 64
    CH = 3
    print(fn)
    if fn != "":
        model = load_model('drivermodel.h5')
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img)

        img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        img = img.astype('float32')
        img = img / 255.0
        print('img shape:', img)
        prediction = model.predict(img)
        print(np.argmax(prediction))
        y = np.argmax(prediction)
        print(y)

        categories = [
            "Other Activity",
            "Safe Driving",
            "Talking Phone",
            "Texting Phone",
            "Turning",
            
        ]

        if y < len(categories):
            Cd = categories[y]
        else:
            Cd = None

    if Cd:
        return Cd
    else:
        raise ValueError("Uploaded image does not match any valid military vehicle category.")
        
        
        

def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=300, y=450)
# def train_model():
    
#     update_label("Model Training Start...............")
    
#     start = time.time()

#     X=Model_frm.main()
    
#     end = time.time()
        
#     ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
#     msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

   # update_label(msg)
def send_email(result):
    sender_email = "pragati.code@gmail.com"
    sender_password = "grqheqzoutabdfzd"
    recipient_email = "ankita.sctcode@gmail.com"
    subject = "Military Vehicle Detection Result"
    body = f"Result: {result}"

    try:
      msg = MIMEMultipart()
      msg['From'] = sender_email
      msg['To'] = recipient_email
      msg['Subject'] = subject
      msg.attach(MIMEText(body, 'plain'))

      with SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            update_label("Result sent via email successfully!")
    except Exception as e:
        update_label(f"Failed to send email: {e}")


def test_model():
    global fn
    if fn != "":
        update_label("Model Testing Start...............")
        
        start = time.time()
        try:
            X = test_model_proc(fn)
            X1 = "{0} ".format(X)
            msg = f"Image Testing Completed..\n{X1}\nExecution Time: {time.time() - start:.4f} seconds"
        except ValueError as e:
            msg = f"Error: {e}"
    else:
        msg = "Please Select Image For Prediction...."

    update_label(msg)
    send_email(msg)

    
def openimage():
   
    global fn
    fileName = askopenfilename(initialdir='C:/Users/admin/Downloads/militry code', title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])



    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root, image=imgtk, height=250, width=250)
    

    img.image = imgtk
    img.place(x=300, y=100)
   # out_label.config(text=imgpath)

def convert_grey():
    global fn    
    IMAGE_SIZE=200
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250, font=("bold", 25), bg='bisque2', fg='black',height=250)
    #result_label1.place(x=300, y=400)
    img2 = tk.Label(root, image=imgtk, height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250, font=("bold", 25), bg='bisque2', fg='black')
    #result_label1.place(x=300, y=400)


def window():
    root.destroy()

button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button1.place(x=10, y=10)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button2.place(x=10, y=70)

button4 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
button4.place(x=10, y=130)

# button4 = tk.Button(frame_alpr, text="Train model", command=train_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
# button4.place(x=10, y=190)


exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '),bg="Green",fg="white")
exit.place(x=10, y=260)



root.mainloop()