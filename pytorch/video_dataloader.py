from torch.utils.data import DataLoader, Dataset
import h5py

class H5VideoDataset(Dataset):
    def __init__(self, h5_file, transforms=None):
        """
        initialzie the dataset from the give h5 file
        args:
            h5_file: the h5 file containing the images to be processed
        """
        self.h5file = h5py.File(h5_file,'r')
        self.dset_name = [name for name in self.h5file['data']][0]

    def __len__(self):
        """
        return the length of the file
        """
        return self.h5file['data'][self.dset_name].shape[0]

    def __getitem__(self, idx):
        """
        returns the item at the given index
        args:
            idx: the index of the item to return
        """
        return self.h5file['data'][self.dset_name][idx]