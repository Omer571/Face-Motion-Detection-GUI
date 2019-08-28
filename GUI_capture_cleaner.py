#CONTROL + ALT + N TO START
import shutil
import numpy.core.multiarray
import cv2, time
import os, pandas
from datetime import datetime
from tkinter import *
import tkinter.messagebox
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource

class GUI_capture(object):
    def __init__(self, window):
        self.window = window
        self.window.wm_title("Website Blocker")

        run_button = Button(window, height = 2, width = 12,text= "Run Program", command = self.run_command) 
        run_button.grid(row = 0, column = 0)

        self.function = IntVar()

        x = Label(window, text = "Pick a function", justify = LEFT, padx = 20)
        x.grid(row = 0, column = 1)
        y = Radiobutton(window, text = "Face Detection", padx = 20, variable = self.function, value = 1, command = self.face_detection_choice)
        y.grid(row = 1, column = 1,)
        z = Radiobutton(window, text = "Motion Detection", padx = 20, variable = self.function, value = 2, command = self.motion_detection_choice)
        z.grid(row = 2, column = 1)
   
    def face_detection_choice(self):
        self.check_box = "FD"
   
    def motion_detection_choice(self):
        self.check_box = "MD"
   
    def run_command(self):
        #### IF FACEBOX CHECKED ####
        if self.check_box[0] == 'f' or self.check_box[0] == 'F':
            
            #if faces directory already existed, delete its contents
            path = os.getcwd() 
            try:
                shutil.rmtree(path + "/faces")
                print("\nOld '/faces' tree deleted\n")
            except:
                print("\nNo old 'faces' directory to delete\n")

            path = path + "/faces"

            try:  
                os.mkdir(path)
            #if file already exists
            except OSError: 
                print ("\nTO ACCESS THE FACES OF THOSE FOUND ON WEBCAM, GO TO %s \n" % path)
            else:  
                print("\nDirectory created at:")
                print ("TO ACCESS THE FACES OF THOSE FOUND ON WEBCAM, GO TO %s \n" % path)

            frames = 1
            img_counter = 0
            images = []
            status_list = [0, 0]
            times = []

            #start video
            video = cv2.VideoCapture(0) 
            #import file with face features for comparison
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            
            df = pandas.DataFrame(columns=["Start", "End"])
            print("PRESS Q TO QUIT")
            while True:
                status = 0
                frames = frames + 1
                # check is a boolean, frame is numpy representation of first video frame
                check, frame = video.read()

                faces = face_cascade.detectMultiScale(frame, 
                scaleFactor = 1.1, 
                minNeighbors = 5) 

                #if faces in frame and faces weren't previously in frame, snapshot it
                if faces != ():
                    img_counter = img_counter + 1
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(os.path.join(path , img_name), frame)
                    images.append(img_name)
                    status = 1

                for x, y, w, h in faces:        
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3) # blue=0, green=255, red=0, thickness = 3
                
                # add status of motion in current frame
                status_list.append(status)
                
                # to avoid to much memory used over hours (use only last two statuses of video)
                status_list = status_list[-2:]

                # record when change in motion
                if status_list[-1] == 1 and status_list[-2] == 0:
                    times.append(datetime.now())
                if status_list[-1] == 0 and status_list[-2] == 1:
                    times.append(datetime.now())

                cv2.imshow("Capturing", frame)

                # 1 frame / 1 ms
                # NOTE: FPS is not actually 1 fpms, program must also run
                key = cv2.waitKey(100)

                # if key pressed was q
                if key == ord('q'):
                    break

            for i in range(0, len(times), 2):
                # record start and end times and ignore 
                try:
                    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index = True)
                except:
                    pass

            video.release()
            cv2.destroyAllWindows()
            message ="To look at all faces that were within frame during program, go to:\n " + path
            tkinter.messagebox.showinfo('Notification', message)
            
        #### IF MOTION BOX CHECKED, JUST RUN MOTION DETECTOR ####
        elif self.check_box[0] == 'M' or self.check_box[0] == 'm':
            first_frame = None
            frame_num = 0
            video = cv2.VideoCapture(0) 
            status_list = [0, 0]
            times = []

            self.df = pandas.DataFrame(columns=["Start", "End"])
            print("PRESS Q TO QUIT")
            while True:
                
                frame_num = frame_num + 1
                
                if frame_num == 20:
                    #refresh screen
                    first_frame = None
                    frame_num = 0
                
                # check is a boolean, frame is numpy representation of first video frame
                check, frame = video.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                status = 0

                if first_frame is None:
                    first_frame = gray
                    continue

                # compares first frame with new frame
                delta_frame = cv2.absdiff(first_frame, gray)

                # make threshold limit ( if > 30 -> white (255))
                # returns tuple (first value for other threshold methods, second is fram thats returned)
                threshold_frame = cv2.threshold(delta_frame, 30 ,255, cv2.THRESH_BINARY)[1]

                # delete white area (motion) after motion gone
                threshold_frame = cv2.dilate(threshold_frame, None, iterations = 2)

                # retrieve external contours
                # chain is method which retrieves
                (cnts,_) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # keep only areas with more than say 10000 pixels
                for contour in cnts:
                    if cv2.contourArea(contour) < 10000:
                        continue
                    status = 1

                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3) # blue=0, green=255, red=0, thickness = 3
                
                # add status of motion in current frame
                status_list.append(status)
                
                # to avoid to much memory used over hours (use only last two statuses of video)
                status_list = status_list[-2:]

                # record when change in motion
                if status_list[-1] == 1 and status_list[-2] == 0:
                    times.append(datetime.now())
                if status_list[-1] == 0 and status_list[-2] == 1:
                    times.append(datetime.now())
                
                cv2.imshow("Color Frame", frame)

                key = cv2.waitKey(1)

                # if key pressed was q
                if key == ord('q'):
                    if status == 1:
                        times.append(datetime.now())
                    break

            # iterate through times list and add to pandas df
            for i in range(0, len(times), 2):
                # record start and end times and ignore 
                self.df = self.df.append({"Start": times[i], "End": times[i+1]}, ignore_index = True)

            # can access in format csv in excel
            self.df.to_csv("Times.csv")
 
            video.release()
            cv2.destroyAllWindows()

            #ASK IF THEY WOULD LIKE TO PRINT GRAPH OF MOTION HERE
            answer = tkinter.messagebox.askquestion('Question 1','Would you like to display a graph of the motion that occured?')
            if answer == "Yes" or answer == "yes" or answer == "Y" or answer == "y":
                
                #format dataframe times so that its readable (before is was 1.52e12)
                self.df["Start_string"] = self.df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
                self.df["End_string"] = self.df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")

                p=figure(x_axis_type="datetime", height = 100, width = 500, sizing_mode='scale_both', title="Motion Graph")

                #we don't need y-axis tick
                p.yaxis.minor_tick_line_color=None

                #dont need intermediete y-axis lines across graph
                p.ygrid[0].ticker.desired_num_ticks=1

                #data in HoverTool
                cds=ColumnDataSource(self.df)

                #hover tool pop window
                hover=HoverTool(tooltips=[("Start", "@Start_string"), ("End ", "@End_string")])
                p.add_tools(hover)

                q=p.quad(left="Start", right = "End", bottom=0,top=1,color="Green", source=cds)

                output_file("Graph.html")
                show(p)


window = Tk()
GUI_capture(window)
window.mainloop()