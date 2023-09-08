import os
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.network_main import ProtoNet
from dataloader.fsl_dataset import FSLDataset, batch_test_task_collate
from model.get_acc import cal_cfm


def train(args):
    print(args)
    #创建相关路径
    exp_dir = Path('./train_log/')
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


    checkpoints_dir = exp_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    log = open(os.path.join(exp_dir, args.log_name + '.txt'), 'w+')

    #准备数据集
    train_dataset = FSLDataset(args, mode='train', episode=args.train_episo, data_augmentataion=True)
    print('meta-training episodes:', len(train_dataset))
    valid_dataset = FSLDataset(args, mode='valid', episode=args.valid_episo, data_augmentataion=False)
    print('validating episodes:', len(valid_dataset))

    #创建dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, collate_fn=batch_test_task_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4, collate_fn=batch_test_task_collate)

    #初始化模型
    model = ProtoNet(args)

    #创建优化器和自动调整学习率
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = model.cuda()

    #开始训练
    best_instance_acc = 0
    for epoch in range(args.nepoch):
        model.train()
        acc = []
        total_loss = []
        test_total_loss = []
        test_acc = []
        loop = tqdm(enumerate(train_loader, 0), total=len(train_dataset), smoothing=0.9)
        for i, data in loop:
            optimizer.zero_grad()
            [support_x, _, query_x, query_y] = data
            support_x = support_x.cuda()
            query_x = query_x.cuda()
            query_y = query_y.cuda()
            pred, loss = model(support_x, query_x, query_y)

            total_loss.append(loss.cpu().detach().numpy())

            cfm = cal_cfm(pred, query_y.view(-1), ncls=args.n_way)
            batch_acc = np.trace(cfm) / np.sum(cfm)
            acc.append(batch_acc)

            # pred_choice = pred.data.max(2)[1]
            # correct = pred_choice.eq(query_y).cpu().sum()
            # acc.append(correct.item() / float(args.n_way * args.q_query))

            loss.backward()
            optimizer.step()

            loop.set_description('Epoch [{}/{}]'.format(epoch + 1, args.nepoch))
            loop.set_postfix(loss=np.mean(total_loss), acc=np.mean(acc))
        scheduler.step()

        with torch.no_grad():
            model.eval()
            valid_loop = tqdm(enumerate(valid_loader, 0), total=len(valid_dataset), smoothing=0.9)
            for j, data in valid_loop:
                [valid_support_x, _, valid_query_x, valid_query_y] = data
                valid_support_x = valid_support_x.cuda()
                valid_query_x = valid_query_x.cuda()
                valid_query_y = valid_query_y.cuda()
                pred, loss = model(valid_support_x, valid_query_x, valid_query_y)

                test_total_loss.append(loss.cpu().detach().numpy())

                cfm = cal_cfm(pred, valid_query_y.view(-1), ncls=args.n_way)
                batch_acc = np.trace(cfm) / np.sum(cfm)
                test_acc.append(batch_acc)

                # pred_choice = pred.data.max(2)[1]
                # correct = pred_choice.eq(valid_query_y).cpu().sum()
                # test_acc.append(correct.item() / float(args.n_way * args.q_query))

                valid_loop.set_description('Test')
                valid_loop.set_postfix(loss=np.mean(test_total_loss), acc=np.mean(test_acc))

            test_instance_acc = np.mean(test_acc)

            test_accuracies = np.array(test_acc)
            test_accuracies = np.reshape(test_accuracies, -1)
            stds = np.std(test_accuracies, 0)
            std = 1.96 * stds / np.sqrt(args.test_episo)

            if (test_instance_acc >= best_instance_acc):
                best_instance_acc = test_instance_acc
                if args.save_model:
                    torch.save(model.state_dict(), os.path.join(checkpoints_dir, args.log_name + '.pth'))
            log.write('Epoch [{}/{}] The Instance Accuracy of Test: {:.2%}, std: {:.2%}\n'.format(epoch + 1, args.nepoch, test_instance_acc, std))
    log.write('-------------------------------------------------------------------\n')
    log.write('The Highest Instance Accuracy is:{}\n'.format(best_instance_acc))
    log.close()



