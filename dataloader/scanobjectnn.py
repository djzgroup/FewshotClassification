import torch.utils.data as data
import os
import os.path

class ScanObjectNNDataset(data.Dataset):
    def __init__(self, root):
        super(ScanObjectNNDataset, self).__init__()
        self.root = root
        self.classfile = os.path.join(self.root, 'shape_names.txt')
        self.testfile = os.path.join(self.root, 'split1.txt')
        self.class_names = [line.strip() for line in open(self.classfile)]

        self.cat2class = {i: name for i, name in enumerate(self.class_names)}
        print(self.cat2class)
        self.class2cat = {self.cat2class[i]: i for i in self.cat2class}

        self.split = [line.strip() for line in open(self.testfile)]  # names of test classes

        self.test_class = [self.class2cat[i] for i in self.split]
        self.train_class = [self.class2cat[i] for i in self.class_names if i not in self.split]

        self.train_data_path = {self.class2cat[i]: os.path.join(self.root, i) for i in
                                list(set(self.class_names) - set(self.split))}
        self.test_data_path = {self.class2cat[i]: os.path.join(self.root, i) for i in self.split}


if __name__ == '__main__':
    root = '../dataset/ScanObjectNN/PB_T50_RS_nobg_txt'
    dataset = ScanObjectNNDataset(root)
    print(dataset.class_names)
    print(dataset.split)
    print(list(set(dataset.class_names) - set(dataset.split)))
    print(dataset.train_data_path)
    print(dataset.test_data_path)
    print(dataset.train_class)
    print(dataset.test_class)
