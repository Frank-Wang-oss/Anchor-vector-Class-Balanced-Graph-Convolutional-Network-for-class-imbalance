import h5py
import torch as tr
import numpy as np
import random
import os
import PIL.Image as Image
import torchvision as tv

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    tr.backends.cudnn.deterministic = True

class data_loader():
    def __init__(self, args):
        self.args = args
        path = './textile/matchingtDATASET_train_32.h5'
        self.num_test = 100
        self.imb_factor = 0.5
        self.trans_train = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5890],
                                    [0.1989])
        ])
        self.train, self.test, self.tlabel, self.num_cls = self.read(path)


    def read(self, path):
        np.random.seed(5)
        data_dict = {}
        train = {}
        test_data = []
        test_label = []
        imb_cls = self.get_imb_cls()
        print(imb_cls)
        with h5py.File(path, mode = 'r') as F:
            for ii, k in enumerate(F.keys()):
                print(k)
                cls = F[k]
                for i, cls_k in enumerate(cls.keys()):
                    if i == 0:
                        data = cls[cls_k][:]
                        data_dict[ii] = data

        for k in data_dict.keys():
            data_cls = []
            data_ = data_dict[k]
            np.random.shuffle(data_)
            for i, m in enumerate(data_):
                if i < 5:
                    self.save_sample(np.squeeze(m), k,i)
            for j in data_:
                img = self.img_process(np.squeeze(j))
                data_cls.append(img)
            data_cls = tr.stack(data_cls, 0)
            train[k] = data_cls[:imb_cls[k]]
            test_data.extend(data_cls[-self.num_test:])
            test_label.append(tr.LongTensor([k]).repeat(self.num_test))

        test_data = tr.stack(test_data, 0)
        test_label = tr.cat(test_label, 0)
        print(test_data.size())
        print(test_label.size())
        return train, test_data, test_label, imb_cls

    def get_imb_cls(self):
        max_num = 1500-self.num_test
        imb_cls = []
        for i in range(6):
            imb_cls.append(int(max_num*(self.imb_factor**i)))

        return imb_cls

    def save_sample(self, img, cls, i):
        print(img)
        img=img*350
        img = Image.fromarray(img)
        img = img.resize((256, 256), Image.ANTIALIAS)

        print(img)
        img = img.convert('RGB')
        print(np.array(img))

        img.save('./{}_{}.jpg'.format(cls, i))

    def img_process(self, img):
        img = Image.fromarray(img)
        img = img.resize((self.args.Image_size, self.args.Image_size), Image.ANTIALIAS)
        img = self.trans_train(img)

        return img


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
    data_loader(args)
