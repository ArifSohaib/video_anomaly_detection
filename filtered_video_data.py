"""tests video averaging"""
import argparse
import cv2
import scipy.misc
import pickle
import h5py
import numpy as np

#construct the argumentpase and parse th arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="Path to input video file")
ap.add_argument("-o", "--output", required=True,
                help="path to output file")
ap.add_argument("-ow", "--output_width", required=False, 
                help="width of output")
ap.add_argument("-oh", "--output_height",required=False,
                 help="height of output")
args = vars(ap.parse_args())

#initialize the Red, green and blue averages
(rAvg, gAvg, bAvg) = (None, None, None)
#initialize the total number of frames read from the file
total_frames = 0

#load the saved impact frames dict
with open('./data/frames/frames_vid1.pkl', 'rb') as f:
    frames = pickle.load(f)
    frames = sorted(list(frames.values()))
#use a counter to keep track of which impact frame is being processed
impact_frame_idx = 0

#open a pointer to the video file
print("[INFO] opening video file pointer...")
stream = cv2.VideoCapture(args["video"])
num_frames = stream.get(cv2.CAP_PROP_FRAME_COUNT)

#open a pointer to h5 file for writing
h5f = h5py.File(args["output"], "w")

width=227
height=227
try:
    if args["output_width"] != None:
        width = int(args["output_width"])
    if args["output_height"] != None:
        height = int(args["output_height"])
except:
    print("[ERROR]please enter an integer for output_height and output_width")
dset = h5f.create_dataset(name=args["video"][:-4]+"_smaller", shape=(num_frames/5, height, width,3), dtype=np.uint8)
print("[INFO] computing frame averages, this may take a while...")
#current index being processed
current_idx = impact_frame_idx
output_frame = 0
#loop over the frames from the video file stream
while True:
    if total_frames == num_frames:
        break
    #grab the frame from the file stream
    (grabbed, frame) = stream.read()
    #if a frame was not grabbed, we have reached the end of the file
    if grabbed is False:
        break

    #split the frame into its respective channels
    (B, G, R) = cv2.split(frame.astype("float"))

    #if the frame averages are none, initialize them
    if rAvg is None:
        rAvg = R
        gAvg = G
        bAvg = B
    #otherwise compute the weighed average between the history of frames and current frame
    else:
        rAvg = ((total_frames * rAvg) + (1 * R)) / (total_frames + 1.0)
        gAvg = ((total_frames * gAvg) + (1 * G)) / (total_frames + 1.0)
        bAvg = ((total_frames * bAvg) + (1 * B)) / (total_frames + 1.0)

    #increment the total number of frames read thus far
    total_frames += 1

    #for every 3 frames, get the averaged output for the FCN
    try:
        if total_frames == frames[impact_frame_idx] - 30:
            #increment the impact frame counter, this should only be done the first time
            print("start of skiped frames at {}".format(total_frames))
            current_idx = impact_frame_idx
            impact_frame_idx += 1
            #check if the frame is within +-30 of the previous skipped frame
        elif (frames[current_idx] - 30) < total_frames <= (frames[current_idx] + 30):
            print("still skipping {}".format(total_frames))

        elif total_frames % 5 == 0:
            # print("averaging 5 frames")
            #merge the RGB averages together
            avg = cv2.merge([bAvg, gAvg, rAvg]).astype(np.uint8)
            avg = cv2.resize(avg, (width, height))

            #write averaged frames to output video
            dset[output_frame,:] = avg
            output_frame += 1
            #clear the averages
            (rAvg, gAvg, bAvg) = (None, None, None)
    except IndexError:
        avg = cv2.merge([bAvg, gAvg, rAvg]).astype(np.uint8)
        # avg = cv2.resize(avg, (width, height))
        avg = cv2.resize(avg, (width, height))
        dset[output_frame] = avg
        break
#close the file pointer
stream.release()
h5f.close()
