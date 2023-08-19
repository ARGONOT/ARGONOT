from email.mime import image
import tkinter as tk
from app import downloadVideos
from tkinter import filedialog as fd
import cv2
import numpy as np
import glob 
from Youtube import Bot
import os
from info import username,password




root = tk.Tk()
root.title("Tiktok")
root.geometry('1100x800')
canvas = tk.Canvas(root, width=650, height=650)
canvas.pack()
entry1 = tk.Entry(root,width=30,bd=3) 
canvas.create_window(200, 140, window=entry1)

entry2 = tk.Entry(root,width=5,bd=3) 
canvas.create_window(200, 180, window=entry2)

entry3 = tk.Entry(root,width=30,bd=3) 
canvas.create_window(500, 525, window=entry3)

entry4 = tk.Entry(root,width=30,bd=3) 
canvas.create_window(500, 550, window=entry4)

def tagname():  
    tagname = entry1.get()
    count = entry2.get()
    folder = open("hashtag.txt", "w")
    folder.write(tagname)
    foldercount = open("count.txt","w")
    foldercount.write(count)

def account():
    email = entry3.get()
    password = entry4.get()
    foldermail = open("email.txt","w")
    foldermail.write(email)
    folderpassword = open("password.txt","w")
    folderpassword.write(password)

def save():
    filetypes = (
        ('All files', '*.*'),
        ('text files', '*.txt'),
        ('Python Files', '*.py'),
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    foldersave = open("Image.txt", "w")
    foldersave.write(filename)


def addPNG():
    imageName = open("Image.txt").read() 
    img = cv2.imread(imageName)
    path = "./*.mp4"
    files = glob.glob(path)

    for video in files:
        videoName = video.split("\\")[1]
        cap = cv2.VideoCapture(video.split("\\")[1])
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
        width = int(width)
        height = int(height)
        print("video Width: ",width, "Video Height: ",height)
        writer = cv2.VideoWriter("./Videos/"+video.split("\\")[1].split(".")[0]+".mp4" , cv2.VideoWriter_fourcc(*"DIVX"),20,(width,height))
        img_height, img_width, _ = img.shape
        print("Width: ",img_width,"Height: ",img_height)

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame[(int(height*0.98)-img_height) : int(height*0.98), (int(width*0.65)-img_width) : int(width*0.65)] = img
                #cv2.imshow("s",frame)
                writer.write(frame)
            
            else: 
                break
        
        cap.release()
        cv2.destroyAllWindows()

def deleteVideos():
    path1 = "./*.mp4"
    path2 = "./Videos/*.mp4"
    files1 = glob.glob(path1)
    files2 = glob.glob(path2)

    for file in files1:
        file = file.split("\\")[1]
        os.remove(file)

    for file in files2:
        os.remove(file)



button1 = tk.Button(root,width=20,height=3,background="green",text='Verify Hashtag and Count', command=tagname)
canvas.create_window(200, 360, window=button1)

Button = tk.Button(width=20,height=3,background="red",text='Download Videos', command=downloadVideos)
canvas.create_window(200, 420, window=Button)

btn = tk.Button(root,width=20,height=3,background="orange" ,text='Select PNG', command=lambda: save())
canvas.create_window(200, 480, window=btn)


buttonPNG = tk.Button(root,width=20,height=3,background="pink" ,text='Add PNG', command=lambda:addPNG())
canvas.create_window(200, 540, window=buttonPNG)

buttonYoutube = tk.Button(root,width=20,height=3,background="red" ,text='Upload Youtube', command=lambda:Bot(username,password).videoUpload())
canvas.create_window(200, 600, window=buttonYoutube)

buttonAccount = tk.Button(root,width=20,height=3,background="yellow" ,text='Verify email-pass', command=lambda:account())
canvas.create_window(500, 600, window=buttonAccount)

buttonDelete = tk.Button(root,width=20,height=3,background="pink" ,text='Delete Videos', command=lambda:deleteVideos())
canvas.create_window(200, 660, window=buttonDelete)


root.mainloop()