import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.get_acc import cal_cfm
from model.network_main import ProtoNet
from dataloader.fsl_dataset import FSLDataset, batch_test_task_collate

def test(args):
    print(args)
    # 创建相关路径
    exp_dir = Path('./eval_log/')
    exp_dir.mkdir(exist_ok=True)
    if args.dataset == 'modelnet40':
        exp_dir = exp_dir.joinpath('modelnet40')
        exp_dir.mkdir(exist_ok=True)
    elif args.dataset == 'scanobjectnn':
        exp_dir = exp_dir.joinpath('scanobjectnn_PB_T50_RS_nobg')
        exp_dir.mkdir(exist_ok=True)
        if args.split_number == 0:
            exp_dir = exp_dir.joinpath('S_0')
            exp_dir.mkdir(exist_ok=True)
        elif args.split_number == 1:
            exp_dir = exp_dir.joinpath('S_1')
            exp_dir.mkdir(exist_ok=True)
        elif args.split_number == 2:
            exp_dir = exp_dir.joinpath('S_2')
            exp_dir.mkdir(exist_ok=True)


    log = open(os.path.join(exp_dir, args.log_name + '.txt'), 'w+')

    # 准备数据集
    test_dataset = FSLDataset(args, mode='test', episode=args.test_episo, data_augmentataion=False)
    print('testing episodes:', len(test_dataset))

    # 创建dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, collate_fn=batch_test_task_collate)

    # 初始化模型
    model = ProtoNet(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()

    # 开始测试
    test_total_loss = []
    test_acc = []
    with torch.no_grad():
        model.eval()
        valid_loop = tqdm(enumerate(test_loader, 0), total=len(test_dataset), smoothing=0.9)
        for j, data in valid_loop:
            [valid_support_x, _, valid_query_x, valid_query_y] = data
            valid_support_x = valid_support_x.cuda()
            valid_query_x = valid_query_x.cuda()
            valid_query_y = valid_query_y.cuda()
            pred, loss = model(valid_support_x, valid_query_x, valid_query_y)

            test_total_loss.append(loss.cpu().detach().numpy())

            # pred_choice = pred.data.max(2)[1]
            # correct = pred_choice.eq(valid_query_y).cpu().sum()
            # test_instance_acc = correct.item() / float(args.n_way * args.q_query)
            # test_acc.append(test_instance_acc)

            cfm = cal_cfm(pred, valid_query_y.view(-1), ncls=args.n_way)
            batch_acc = np.trace(cfm) / np.sum(cfm)
            test_acc.append(batch_acc)


            valid_loop.set_description('Test')
            valid_loop.set_postfix(loss=np.mean(test_total_loss), acc=np.mean(test_acc))

    test_avg_acc = np.mean(test_acc)
    std_acc = np.std(test_acc)
    interval = 1.960 * (std_acc / np.sqrt(len(test_acc)))
    print('The Average Instance Accuracy is: {}, Interval: {}'.format(test_avg_acc*100, interval*100))
    log.write('The Average Instance Accuracy is: {}, Interval: {}\n'.format(test_avg_acc*100, interval*100))
    log.close()


