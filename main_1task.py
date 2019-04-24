import argparse
import os
from dataset.dataset_1task import get_loader
from solver_1task import Solver


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config.batch_size, num_thread=config.num_thread)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_fold, run)): run += 1
        os.mkdir("%s/run-%d" % (config.save_fold, run))
        os.mkdir("%s/run-%d/models" % (config.save_fold, run))
        config.save_fold = "%s/run-%d" % (config.save_fold, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader = get_loader(config.batch_size, mode='test',num_thread=config.num_thread, sal_mode=config.sal_mode)
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':

    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5) # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--resnet', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=1) # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=3)

    # Testing settings
    parser.add_argument('--model', type=str, default=None) # Snapshot
    parser.add_argument('--test_fold', type=str, default=None) # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='e') # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
