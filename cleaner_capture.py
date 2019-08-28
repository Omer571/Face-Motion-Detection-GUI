#CONTROL + ALT + N TO START
import shutil
import cv2, time
import os, pandas
from datetime import datetime
facebox = True

#### IF FACEBOX CHECKED ####
if facebox == True:
    
    #if faces directory already existed, delete its contents
    path = os.getcwd() 
    try:
        shutil.rmtree(path + "/faces")
        print("Old '/faces' tree deleted\n")
    except:
        print("No old 'faces' directory to delete\n")
    #print("\n\nTHE CURRENT WORKING DIRECTORY IS %s" % path) 

    path = path + "/faces"

    try:  
        os.mkdir(path)
    #if file already exists
    except OSError: 
        print ("\nTO ACCESS THE FACES OF THOSE FOUND ON WEBCAM, GO TO %s \n" % path)
        #print ("\nFailed to make directory %s to hold captured images\n" % path)
    else:  
        print("\nDirectory created at:")
        print ("TO ACCESS THE FACES OF THOSE FOUND ON WEBCAM, GO TO %s \n" % path)

    frames = 1
    img_counter = 0
    images = []
    status_list = [0, 0]
    times = []

    #start video
    video = cv2.VideoCapture(0) # pass parameter if you have video, else pass 0 for webcam
    #import file with face features for comparison
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #make data frame with start and end columns
    df = pandas.DataFrame(columns=["Start", "End"])

    print("PRESS Q TO QUIT")
    while True:
        status = 0
        frames = frames + 1
        # check is a boolean, frame is numpy representation of first video frame
        check, frame = video.read()

        #print(check)
        #print(frame)
        #gray version (better because don't have to worry about lighting)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 
        scaleFactor = 1.1, # decreases scale by 10% every search and searches for bigger image
        minNeighbors = 5) 
        #print(faces)

        #if faces in frame and faces weren't previously in frame, snapshot it
        if faces != ():
            #print("detection\n\n\n\n\n\n\n\n\n\n\n\n")
            img_counter = img_counter + 1
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(os.path.join(path , img_name), frame)
            print("{} written!".format(img_name))
            images.append(img_name)
            status = 1

        for x, y, w, h in faces:# top-left bottom-right         
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3) # blue=0, green=255, red=0, thickness = 3
        
        # add status of motion in current frame
        status_list.append(status)
        
        # to avoid to much memory used over hours (use only last two statusus of video)
        # NOTE: this doesn't help overall program purpose
        status_list = status_list[-2:]

        # record when change in motion
        if status_list[-1] == 1 and status_list[-2] == 0:
            times.append(datetime.now())
        if status_list[-1] == 0 and status_list[-2] == 1:
            times.append(datetime.now())

        # wait 3 seconds
        #time.sleep(3)
        
        # show that first captured frame
        cv2.imshow("Capturing", frame)

        # 1 frame / 1 ms
        # NOTE: FPS is not actually 1 fpms, program must also run
        key = cv2.waitKey(100)

        # if key pressed was q
        if key == ord('q'):
            break

        #print(flag)
    #print(times)

    for i in range(0, len(times), 2):
        # record start and end times and ignore 
        try:
            df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index = True)
        except:
            pass

    video.release()
    cv2.destroyAllWindows()
