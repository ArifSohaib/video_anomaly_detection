import h5py

h5f = h5py.File('../data/filtered_smaller_period1_227.h5')
for name in h5f['data']:
    break
data = h5f['data'][name][()]
