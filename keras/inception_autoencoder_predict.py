from inception_autoencoder import build_autoencoder
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing

def main():

    impact_data = np.loadtxt('../data/features/impacts_period1.npy')
    full_data = np.loadtxt('../data/features/period1_full.npy')
    min_max_scaler = preprocessing.MinMaxScaler()
    impact_data = min_max_scaler.fit_transform(impact_data)
    full_data = min_max_scaler.fit_transform(full_data)

    model = build_autoencoder()
    model.load_weights('../data/weights/autoencoder_weights.h5')

    predict_impact = model.predict(impact_data)
    predict_full = model.predict(full_data)

    mse_full = ((predict_full - full_data) ** 2).mean(axis=1)
    mse_impact = ((predict_impact - impact_data) ** 2).mean(axis=1)
    mean_mse = mse_full.mean()
    print("full mse avg {}".format(mean_mse))
    print("impact mse avg {}".format(mse_impact.mean()))

    print("full mse min {}".format(mse_full.min()))
    print("impact mse min {}".format(mse_impact.min()))

    print("full mse max {}".format(mse_full.max()))
    print("impact mse max {}".format(mse_impact.max()))

    plt.hist(mse_full, label='full_mse stats')
    plt.show()
    plt.hist(mse_impact, label='impact_mse stats')
    plt.show()

    full_percentile = np.percentile(mse_full, 50)
    impact_percentile = np.percentile(mse_impact, 50);
    print("full mse percentile {}".format(full_percentile))
    print("impact mse percentile {}".format(impact_percentile))
    
    print("length of full data {}".format(len(mse_full)))
    pred_impact_idx = []
    #running the above statistics, we can say that if the mse is above the max of impact mse, then it is not an impact
    for idx, err in enumerate(mse_full):
        if err > impact_percentile:
            pred_impact_idx.append(idx)

    
    with open('../data/frames/frames_vid1_div5.pkl', 'rb') as f:
        confirmed_idx = pickle.load(f)
    confirmed_idx = sorted(confirmed_idx.values())
    """
    for each value in confirmed_idx we need 10 numbers before and 10 after it(totaling 29 * 20 = 580)
    """
    full_idx = []
    for idx in confirmed_idx:
        for i in range(-10, 10):
            full_idx.append(idx+i)

    true_count = 0
    false_pos = 0
    #to check accuracy, we can compare against idx's computed before
    idx_count = 0;
    for idx in pred_impact_idx:
        if idx in full_idx:
            true_count += 1
            
        else:
            false_pos += 1
    print("num predictions {}".format(len(pred_impact_idx)))
    print("true count {}".format(true_count))
    print("length of pred impacts {}".format(len(full_idx)))
    print("false pos {}".format(false_pos))
    
if __name__ == '__main__':
    main()
