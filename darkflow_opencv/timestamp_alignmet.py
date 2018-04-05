import pandas as pd
true_data = pd.read_csv("vid1_timestamps.txt",header=None)

result_data = pd.read_csv("timestamps0.6_sec.txt",header=None)

true_data = list(true_data[0].values)
result_data = list(result_data[0].values)

from datetime import datetime, timedelta

true_data = list(map(lambda x:datetime.strptime(x, "%M:%S"),true_data))
result_data = list(map(lambda x:datetime.strptime(x,"%M:%S"),result_data))

count = 0

true_pos = []
approx_pos = []
for result_time in result_data:
    for true_time in true_data:
        tdelta = abs(true_time-result_time)
        diff = timedelta(days = 0, seconds=tdelta.seconds, microseconds = tdelta.microseconds)
        diff = diff.total_seconds()
        if diff < 2.0:
            count += 1
            true_pos.append(true_time)
            approx_pos.append(result_time)
            #print("real time: {}, calc time: {}".format(true_time, result_time))
        
print("first glance results for 1 threshold")
print("total results: {},\tconfirmed:{}".format(len(result_data), len(true_data)))
print("true positives: {},\tfalse positives:{},\tfalse negatives:{}".format(count,len(result_data)-count,len(true_data)-count))
