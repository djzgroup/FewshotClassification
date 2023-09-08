import glob
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from dataloader.modelnet40 import ModelNetDataset
from dataloader.scanobjectnn import ScanObjectNNDataset

class FSLDataset(data.Dataset):
    def __init__(self, args, mode='train', episode=600, data_augmentataion=False):
        super(FSLDataset, self).__init__()
        self.dataset_type = args.dataset
        if self.dataset_type == 'modelnet40':
            self.dataset = ModelNetDataset(args.data_path)
        elif self.dataset_type == 'scanobjectnn':
            self.dataset = ScanObjectNNDataset(args.data_path)

        self.mode = mode
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.q_query = args.q_query
        self.num_episode = episode
        self.num_point = args.pc_npts
        self.use_norm = args.use_normals
        self.data_augmentation = data_augmentataion

        if mode == 'train':
            self.classes = self.dataset.train_class
        elif mode == 'test' or mode == 'valid':
            self.classes = self.dataset.test_class
        else:
            raise NotImplementedError('Unkown mode: %s!' % mode)
        print(self.classes)
        self.train_data_path = self.dataset.train_data_path
        self.test_data_path = self.dataset.test_data_path


    def __len__(self):
        return self.num_episode


    def __getitem__(self, index):
        sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)
        support_labels = sampled_classes

        query_labels = np.zeros((self.n_way, self.q_query))
        for i in range(self.n_way):
            for j in range(self.q_query):
                query_labels[i][j] = i

        support_ptclouds, query_ptclouds = self.generate_one_episode(sampled_classes)
        if self.data_augmentation:
            support_ptclouds = self.data_augment(support_ptclouds)
            query_ptclouds = self.data_augment(query_ptclouds)


        return support_ptclouds, support_labels, query_ptclouds, query_labels


    def generate_one_episode(self, sampled_classes):
        support_ptclouds = []
        query_ptclouds = []

        for sampled_class in sampled_classes:
            if self.mode == 'train':
                data_path = self.train_data_path[sampled_class]
            elif self.mode == 'test' or self.mode == 'valid':
                data_path = self.test_data_path[sampled_class]

            if self.dataset_type == 'modelnet40' or self.dataset_type == 'scanobjectnn':
                files = glob.glob(os.path.join(data_path, '*.txt'))
            else:
                files = glob.glob(os.path.join(data_path, 'points/*.pts'))

            selected_file = np.random.choice(files, self.k_shot + self.q_query, replace=False)
            support_files = selected_file[:self.k_shot]
            query_files = selected_file[self.k_shot:]

            support_ptclouds_one_way = self.sample_pointclouds(support_files)
            query_ptclouds_one_way = self.sample_pointclouds(query_files)

            support_ptclouds.append(support_ptclouds_one_way)
            query_ptclouds.append(query_ptclouds_one_way)

        support_ptclouds = np.stack(support_ptclouds, axis=0)
        query_ptclouds = np.stack(query_ptclouds, axis=0)

        return support_ptclouds, query_ptclouds


    def sample_pointclouds(self, files):
        ptclouds = []
        for file in files:
            if self.dataset_type == 'modelnet40':
                points = np.loadtxt(file, delimiter=',').astype(np.float32)
            else:
                points = np.loadtxt(file).astype(np.float32)
            choice = np.random.choice(len(points), self.num_point, replace=False)
            points = points[choice, :]
            if not self.use_norm:
                points = points[:, 0:3]
            points = self.pc_normalize(points)
            # if self.dataset_type == 'modelnet40':
            #     points = self.pc_normalize(points)
            # if self.dataset_type == 'scanobjectnn' and self.data_augmentation:
            #     points = self.translate_pointcloud(points)
            #     np.random.shuffle(points)

            ptclouds.append(points)

        ptclouds = np.stack(ptclouds, axis=0)
        return ptclouds

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud

    def data_augment(self, point_set):
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        point_set[:, :, :, [0, 2]] = point_set[:, :, :, [0, 2]].dot(rotation_matrix)  # random rotation
        point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        return point_set


def batch_test_task_collate(batch):
    batch_support_ptclouds, batch_support_labels, batch_query_ptclouds, batch_query_labels = batch[0]

    data = [torch.from_numpy(batch_support_ptclouds), torch.from_numpy(batch_support_labels.astype(np.int64)),
                torch.from_numpy(batch_query_ptclouds), torch.from_numpy(batch_query_labels.astype(np.int64))]

    return data