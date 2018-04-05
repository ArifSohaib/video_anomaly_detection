import numpy as np
from glob import glob


def sorttime(timestamp):          
    return int(timestamp.split(":")[0]),int(timestamp.split(":")[1])
framenums = np.load("impact_idx0.6.npy")
timestamps = []

prev_frame = framenums[0]
continous_count = 0

#filter the list to only give impacts lasting more than half a second
new_frames = []
cont_frames = []
prev = 0
cont_count = 0
curr_count = 0
for frame in framenums:
    if frame - prev <3:
        cont_count += 1
        #put all the continous frames in new list
        cont_frames.append(frame)
    elif cont_count > 60:
        print(cont_count)
        #put the last continous frame in a new list
        new_frames.append(frame)
        #reset
        cont_count = 0
        curr_count += 1
    #     cont_frames = []
    prev = frame
print("{} impacts more than 60 frames".format(curr_count) )
print(len(new_frames))

for frame in new_frames:
    minute = (frame-1)//3600
    second = ((frame-1)%3600)//60
    print("{}:{}".format(minute, second))
    timestamps.append("{}:{}".format(minute,second))

timestamps_set = set(timestamps)
timestamps_set = sorted(timestamps_set,key=sorttime)
with open("timestamps0.6_sec.txt","w") as outfile:
    for timestamp in timestamps_set:
        print(timestamp)
        outfile.write(timestamp+"\n")
