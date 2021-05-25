from torch.utils.data import Dataset


class GeeseDataset(Dataset):
    """G-Net Dataset"""

    def __init__(self, buffers):
        self.buffers = buffers

    def __len__(self):
        return len(self.buffers['states'])

    def __getitem__(self, idx):
        return {key: self.buffers[key][idx] for key in self.buffers}

