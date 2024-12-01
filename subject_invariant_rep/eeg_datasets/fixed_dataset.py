import torch


class FixedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        X = {}
        y = []
        subject_ids = []

        for i in range(len(dataset)):
            sample = dataset[i]
            signals = sample[0]
            labels = sample[1]
            ids = sample[2]
            if type(signals) is list:
                for j in range(len(signals)):
                    if j not in X.keys():
                        X[j] = []
                    X[j].append(signals[j])
            else:
                if 0 not in X.keys():
                    X[0] = []
                X[0].append(signals)
            y.append(labels)
            subject_ids.append(ids)

        self.X = X
        self.y = y
        self.subject_ids = subject_ids
        assert len(X[0]) == len(y) == len(subject_ids)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        X = []
        for i in range(len(self.X)):
            X.append(self.X[i][index])
        y = self.y[index]
        subject_ids = self.subject_ids[index]
        return X, y, subject_ids

    def __len__(self):
        return len(self.X[0])

    def dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self, *args, **kwargs)


__all__ = ["FixedDataset"]
