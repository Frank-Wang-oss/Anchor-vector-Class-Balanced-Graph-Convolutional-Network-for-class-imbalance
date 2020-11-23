import torch as tr
from PIL import Image,ImageOps
import torchvision as tv
import numpy as np
import os
import random
import cv2

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    tr.backends.cudnn.deterministic = True

# seed_torch(0)

class data_loader():
    def __init__(self, args):
        self.args = args
        self.image_size = args.Image_size
        path = './NEU-CLS-64/'
        # self.num_test = 150
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.num_min = 200
        self.num_maj = 1589
        # self.trans = tv.transforms.Compose([
        #     tv.transforms.Grayscale(),
        #     tv.transforms.ToTensor()
        # ])
        if args.noise == 0:
            self.trans_train = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5890],
                                        [0.1989])
            ])
            self.trans_test = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5890],
                                        [0.1989])
            ])
        else:
            self.trans_train = tv.transforms.Compose([
                tv.transforms.ToTensor()
            ])
            self.trans_test = tv.transforms.Compose([
                tv.transforms.ToTensor()
            ])
        self.train, self.test, self.tlabel, self.num_cls = self.read(path)
    def read(self, path):
        train = {}
        test = []
        tlabel = []
        np.random.seed(3)
        dict_cls = {
            0: 'cr', 1: 'gg', 2: 'in', 3: 'pa', 4: 'ps', 5: 'rp', 6:'rs',7:'sc',8:'sp'
        }
        minority_single = True
        num_cls_ = []
        for cls in range(len(dict_cls)):
            image_cls = []
            file_name = os.listdir(path+dict_cls[cls])
            np.random.shuffle(file_name)
            num_cls = len(file_name)
            if minority_single:
                if cls == 5:
                    num_train = num_cls - (self.num_min - self.num_train)
                else:
                    num_train = num_cls - 150
            else:
                num_train = num_cls-(self.num_min-self.num_train)
            print(num_train)
            file_train = file_name[:num_train]
            file_test = file_name[num_train:(num_train+self.num_test)]


            num_cls_.append(num_cls)
            ### train set
            for f in file_train:
                img = self.image_process(path+dict_cls[cls]+'/'+f)
                img = self.trans_train(img)
                img = self.AGWN(img)
                image_cls.append(img)
            image_cls = tr.stack(image_cls,0)
            train[cls] = image_cls
            ### test set
            for f in file_test:
                img = self.image_process(path + dict_cls[cls] + '/' + f)
                img = self.trans_train(img)
                img = self.AGWN(img)
                test.append(img)
                tlabel.append(tr.LongTensor([cls]))

        test = tr.stack(test, 0)
        tlabel = tr.cat(tlabel,0)
        print(num_cls_)
        print(test.size())
        print(tlabel.size())

        return train,test,tlabel, num_cls_



    def image_process(self, path):
        img = Image.open(path)
        # img = ImageOps.equalize(img)
        img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        img = np.array(img.convert('L'))
        # noise = np.array(tr.Tensor(img.shape[0],img.shape[1]).normal_(mean=0, std=1))
        # # print(img)
        # img = img + self.args.noise * noise

        return img

    def AGWN(self, img):

        noise = tr.Tensor(img.size()).normal_(mean=0, std=1)
        # print('img is {}'.format(img))
        # print('noise is {}'.format(noise))

        img = img + self.args.noise * noise

        return img
    def compute_mean_std(self, dict_):
        assemble = []
        mean = []
        std = []
        for k,v in dict_.items():
            assemble.append(v)
        assemble = tr.cat(assemble,0)
        channel = assemble.size(1)
        for i in range(channel):
            mean.append(tr.mean(assemble[:,i,:,:]))
            std.append(tr.std(assemble[:,i,:,:]))
        print(mean)
        print(std)
        return mean, std


def graph_train(args, train):
    ### train must be a dict, while prediction maybe a dict or not.
    Nn = args.n_shot
    Nc = args.n_way
    xi = []
    xi_label = []

    xs = []
    xs_label = []
    xl_onehot = []

    for bs in range(args.batch_size):
        ### sample train data
        xs_i = []
        xs_label_i = []
        xl_onehot_i = []
        random_cls = random.sample(range(Nc),1)
        for cls in range(Nc):
            data_class = train[cls]
            num_cls = data_class.size(0)
            if cls == random_cls[0]:
                index = random.sample(range(num_cls), Nn+1)
                xi.append(data_class[index[0]])
                xi_label.append(tr.LongTensor([cls]))

                xs_i.extend(data_class[index[1:]])


            else:
                index = random.sample(range(num_cls), Nn)
                xs_i.extend(data_class[index])

            xs_label_i.append(tr.LongTensor([cls]).repeat(Nn))
            onehot = tr.zeros(Nc)
            onehot[cls] = 1
            xl_onehot_i.append(onehot.repeat(Nn,1))

        index = np.random.permutation(np.arange(Nc * Nn))
        # xs.append(tr.stack(xs_i, 0))
        # xs_label.append(tr.cat(xs_label_i, 0))
        # xl_onehot.append(tr.cat(xl_onehot_i, 0))
        xs.append(tr.stack(xs_i, 0)[index])
        xs_label.append(tr.cat(xs_label_i, 0)[index])
        xl_onehot.append(tr.cat(xl_onehot_i, 0)[index])
    xi = tr.stack(xi, 0)
    xi_label = tr.cat(xi_label, 0)
    xs = tr.stack(xs,0)
    xs_label = tr.stack(xs_label, 0)
    xl_onehot = tr.stack(xl_onehot, 0)
    # print(xs_label)
    return xi, xi_label, xs, xs_label, xl_onehot





def graph_test(args, train, prediction):
    ### train must be a dict, while prediction maybe a dict or not.
    ### mode have 2 choice, which is training, prediction
    Nn = args.n_shot
    Nc = args.n_way
    xi = []

    xs = []
    xs_label = []
    xl_onehot = []

    for bs in range(args.batch_size):
        ### sample train data
        xs_i = []
        xs_label_i = []
        xl_onehot_i = []
        for cls in range(Nc):
            data_class = train[cls]
            num_cls = data_class.size(0)

            index = random.sample(range(num_cls), Nn)

            xs_i.extend(data_class[index])

            xs_label_i.append(tr.LongTensor([cls]).repeat(Nn))
            onehot = tr.zeros(Nc)
            onehot[cls] = 1
            xl_onehot_i.append(onehot.repeat(Nn, 1))

        index = np.random.permutation(np.arange(Nc * Nn))
        # xs.append(tr.stack(xs_i, 0))
        # xs_label.append(tr.cat(xs_label_i, 0))
        # xl_onehot.append(tr.cat(xl_onehot_i, 0))
        xs.append(tr.stack(xs_i, 0)[index])
        xs_label.append(tr.cat(xs_label_i, 0)[index])
        xl_onehot.append(tr.cat(xl_onehot_i, 0)[index])


        xi.append(prediction[bs])

    xi = tr.stack(xi, 0)
    xs = tr.stack(xs, 0)
    xs_label = tr.stack(xs_label, 0)
    xl_onehot = tr.stack(xl_onehot, 0)



    return xi, xs, xs_label, xl_onehot


if __name__ == '__main__':
    from argument import args
    args = args()
    args.num_train = 100
    import matplotlib.pyplot as plt
    data_loader(args)