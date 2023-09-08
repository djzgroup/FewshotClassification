import os
import argparse
from run.meta_training import train
from run.meta_testing import test

os.environ['CUDA_VISIBLE_DEVICES']='0'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='eval', choices=['prototrain', 'protoeval'])
    parser.add_argument('--data_path', type=str, default='./dataset/modelnet40_normal_resampled')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='the dataset to use')
    parser.add_argument('--model_path', type=str, default='./train_log/modelnet40/checkpoints/5_way_1_shot.pth', help='')
    parser.add_argument('--log_name', type=str, default='5_way_1_shot_2', help='the name of log file')
    parser.add_argument('--in_channel', type=int, default=3, help='input channel of points')
    parser.add_argument('--emb_dims', type=int, default=1024, help='the global point feature dimension of dgcnn')
    parser.add_argument('--pc_npts', type=int, default=512, help='Number of input points for Backbone')
    parser.add_argument('--n_way', type=int, default=5, help='Number of sampled classes')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of support examples from each class')
    parser.add_argument('--q_query', type=int, default=10, help='Number of query examples from each class')
    parser.add_argument('--nepoch', type=int, default=80, help='Number of epochs to train for')
    parser.add_argument('--train_episo', type=int, default=400, help='Number of meta-training episodes')
    parser.add_argument('--valid_episo', type=int, default=600, help='Number of validating episodes')
    parser.add_argument('--test_episo', type=int, default=700, help='Number of meta-testing episodes')
    parser.add_argument('--split_number', type=int, default=0, help='the spilt number of scanobjectnn dataset')
    parser.add_argument('--k', type=int, default=20, help='sampled neighbor points of each point in knn')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model')
    parser.add_argument('--use_normals', type=bool, default=False, help='use normals')
    parser.add_argument('--use_bin', type=bool, default=True, help='use the bin pooling')
    parser.add_argument('--use_sup', type=bool, default=True, help='compensate information for prototype feature')
    parser.add_argument('--use_pqf', type=bool, default=True, help='use the proto and query fusion module')

    args = parser.parse_args()

    if args.phase == 'train':
        train(args)
    elif args.phase == 'eval':
        test(args)
    else:
        raise ValueError('Please set correct phase!')


