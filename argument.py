import argparse as arg


def args():
    args = arg.ArgumentParser()

    args.add_argument('--nf_adj', default=96, type = int, help='the hidden number of element in adjacency matrix')
    args.add_argument('--nf_gc', default= 64, type = int, help = 'number of element in hidden layer of Graph convolution')
    args.add_argument('--nf_cnn', default=64, type = int, help = 'number of element in hidden layer of image embedding')
    args.add_argument('--lr',default=1e-3,type = int, help = 'learning rate, alternative choice is 3e-4,1e-3')
    args.add_argument('--num_layer_gc', default=5, type = int, help = 'numebr of layer in graph convolution')
    args.add_argument('--n_way', default=9, type = int, help = 'number of class choosed')
    args.add_argument('--n_shot', default= 3, type = int, help = 'number of samples from each class')
    args.add_argument('--batch_size', default=15, type = int, help = 'number of batch size')
    args.add_argument('--embedding_size', default=96, type = int, help = 'embedding size')
    args.add_argument('--data_seed', default=0, type = int, help = 'random seed')
    args.add_argument('--model_seed', default=1, type=int, help='random seed')
    args.add_argument('--iteration', default=2001, type = int)
    args.add_argument('--interval', default= 100, type = int, help = 'interval of printing loss')
    args.add_argument('--sample_eval', default= 900, type = int)
    args.add_argument('--save_name', default='test', type = str)
    args.add_argument('--Image_size', default=64, type = int)
    args.add_argument('--pre_num', default=5,type = int, help = 'number of prediction')
    args.add_argument('--base_model', default='alexnet',type = str)
    args.add_argument('--num_train',default=50,type=int)
    args.add_argument('--num_test',default=50,type = int)
    args.add_argument('--noise', default=0, type=float)
    args.add_argument('--visual', default=False, type=bool)
    return args.parse_args()


if __name__ == '__main__':
    import numpy as np
    args = args()
    # np.save('./para/{}.npy'.format(args.save_name), args)
    # print(args)
    a = np.load('./para/{}.npy'.format(args.save_name))
    print(a)
