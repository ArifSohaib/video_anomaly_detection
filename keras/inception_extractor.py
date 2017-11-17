"""
Extracts features to use in LSTM model
"""
import numpy as np
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import imageio
# from data_generator import get_impact_frames
from vid_generator import VideoGenerator

def extract_embedding(filename):
    """
    extracts the avg_pool layer of the inception model as it runs prediction
    the resulting embedding can be used for LSTM network which was shown to have high accuracy
    """
    base_model = InceptionV3(weights='imagenet', include_top=True)

    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('avg_pool').output)


    #load the images
    vid = imageio.get_reader(filename)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for _ in tqdm(range(vid.get_meta_data()['nframes'])):
        image = np.expand_dims(vid.get_next_data(), axis=0)
        features = model.predict(image)[0]
        sequence.append(features)

    # Save the sequence.
    np.savetxt("../data/features/period{}-full.npy".format(filename[6]), sequence)

def extract_embedding_impacts(filename, impact_file):
    """
    extracts the avg_pool layer of the inception model as it runs prediction
    the resulting embedding can be used for LSTM network which was shown to have high accuracy
    """
    base_model = InceptionV3(weights='imagenet', include_top=True)

    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('avg_pool').output)


    #load the images
    vid = imageio.get_reader(filename)

    # Now loop through and extract features to build the sequence.
    sequence = []
    impact_counter = 0
    impact_frames = get_impact_frames(impact_file)
    for i in tqdm(range(vid.get_meta_data()['nframes'])):
        image = np.expand_dims(vid.get_next_data(), axis=0)
        if i == impact_frames[impact_counter]:
            features = model.predict(image)[0]
            sequence.append(features)
            impact_counter += 1
    # Save the sequence.
    np.savetxt("./data/features/impacts_period{}.npy".format(filename[-9]), sequence)

def extract_embedding_impacts_h5(filename, impact_file):
    """
    extracts the avg_pool layer of the inception model as it runs prediction
    the resulting embedding can be used for LSTM network which was shown to have high accuracy
    """
    base_model = InceptionV3(weights='imagenet', include_top=True)

    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('avg_pool').output)

    #load the images
    data = VideoGenerator(filename).get_full_impact_data("../data/frames/" + impact_file)
    return_data = []
    for i in tqdm(data):
        image = np.expand_dims(i, axis=0)
        features = model.predict(image)[0]
        return_data.append(features)
    # Save the sequence.
    np.savetxt("../data/features/impacts_period{}.npy".format(filename[-6]), return_data)

def extract_embedding_nonimpacts_h5(filename, impact_file):
    """
    extracts the avg_pool layer of the inception model as it runs prediction
    the resulting embedding can be used for LSTM network which was shown to have high accuracy
    """
    base_model = InceptionV3(weights='imagenet', include_top=True)

    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('avg_pool').output)

    #load the images
    data = VideoGenerator(filename).get_full_impact_data("../data/frames/" + impact_file)
    return_data = []
    for i in tqdm(data):
        image = np.expand_dims(i, axis=0)
        features = model.predict(image)[0]
        return_data.append(features)
    # Save the sequence.
    np.savetxt("../data/features/impacts_period{}.npy".format(filename[-6]), return_data)

def main():
    # extract_embedding_impacts('U18 vs Waterloo Period 3 299.mp4', 'frames_vid3')
    # extract_embedding_impacts('U18 vs Waterloo Period 2 299.mp4', 'frames_vid2')
    extract_embedding(
        '../data/period1_averaged_224.avi')
    # extract_embedding_impacts_h5('../data/filtered_smaller_period1_224.h5','frames_vid1_div5.pkl')

if __name__ == '__main__':
    main()
