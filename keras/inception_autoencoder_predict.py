from inception_autoencoder import build_autoencoder
import numpy as np


def main():

    impact_data = np.loadtxt('../data/features/impacts_period1.npy')
    full_data = np.loadtxt('../data/features/period1_full.npy')
    
    model = build_autoencoder()
    model.load_weights('../data/weights/autoencoder_weights.h5')

    predict_impact = model.predict(impact_data)
    predict_full = model.predict(full_data)

    mse_full = ((predict_full - full_data) ** 2).mean(axis=1)
    mse_impact = ((predict_impact - impact_data) ** 2).mean(axis=1)
    print("full mse avg {}".format(mse_full.mean()))
    print("impact mse avg {}".format(mse_impact.mean()))

    print("full mse min {}".format(mse_full.min()))
    print("impact mse min {}".format(mse_impact.min()))

    print("full mse max {}".format(mse_full.max()))
    print("impact mse max {}".format(mse_impact.max()))

    pred_impact_idx = []
    #running the above statistics, we can say that if the mse is above the max of impact mse, then it is not an impact
    for idx, err in enumerate(mse_full):
        if err > mse_impact:
            pred_impact_idx.append(idx)

    #to check accuracy, we can compare against idx's computed before
    
if __name__ == '__main__':
    main()
