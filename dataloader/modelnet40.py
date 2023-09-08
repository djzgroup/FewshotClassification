import torch.utils.data as data
import os
import os.path

class ModelNetDataset(data.Dataset):
    def __init__(self, root):
        super(ModelNetDataset, self).__init__()
        self.root = root
        self.cat40file = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat10file = os.path.join(self.root, 'meta_test_shape_names.txt')
        self.class_names = [line.strip() for line in open(self.cat40file)]

        self.cat2class = {i : name for i, name in enumerate(self.class_names)}
        print(self.cat2class)
        self.class2cat = {self.cat2class[i] : i for i in self.cat2class}

        self.split = [line.strip() for line in open(self.cat10file)]# names of test classes

        self.test_class = [self.class2cat[i] for i in self.split]
        self.train_class = [self.class2cat[i] for i in self.class_names if i not in self.split]

        self.train_data_path = {self.class2cat[i]: os.path.join(self.root, i) for i in
                           list(set(self.class_names) - set(self.split))}
        self.test_data_path = {self.class2cat[i]: os.path.join(self.root, i) for i in self.split}




if __name__ == '__main__':
    model = ModelNetDataset(root='../dataset/modelnet40_normal_resampled')
    print(model.class_names)
    print(model.split)
    print(list(set(model.class_names) - set(model.split)))
    print(model.train_data_path)
    print(model.test_data_path)
    print(model.train_class)
    print(model.test_class)
