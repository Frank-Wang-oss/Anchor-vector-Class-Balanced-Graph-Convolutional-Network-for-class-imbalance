import torch as tr
from PIL import Image,ImageOps
import torchvision as tv
import numpy as np
import os
import random
import xlrd
import math





class data_loader():
    def __init__(self, args):
        self.args = args
        path = './wood/'
        self.image_size = args.Image_size
        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5071],
                                    [0.2673])
        ])

        self.train, self.test, self.tlabel, self.test_name = self.data_read(path)

    def process_label(self, label_file):
        workbook = xlrd.open_workbook(label_file + 'labels.xlsx')
        booksheet = workbook.sheet_by_index(0)
        name = {}

        for i in range(booksheet.nrows):
            data = booksheet.cell(i, 0)
            data = data.value
            data = str(data)
            data_name = data.split()
            name[data_name[0]] = data_name[1]

        return name

    def data_read(self, path):
        # np.random.seed(self.args.data_seed)
        img_path = path + 'imgs/'
        label_path = path
        n_l = self.process_label(label_path)
        wood_data = {}
        wood_name = {}
        lb = ['dry_knot', 'encased_knot', 'horn_knot', 'decayed_knot', 'leaf_knot', 'edge_knot', 'sound_knot']
        lb_dict = {}

        train = {}
        test_data = []
        test_label = []
        test_name = []
        for i in range(7):
            lb_dict[lb[i]] = i

        for k, v in n_l.items():
            img = Image.open(img_path + k)
            # print(img)
            img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
            # img = img.convert('L')
            img_array = self.transform(img)

            noise = tr.Tensor(img_array.size()).normal_(mean=0, std=1)
            # print(img_array)
            # print(noise)
            img_array = img_array + self.args.noise * noise
            if lb_dict[v] not in wood_data.keys():
                wood_data[lb_dict[v]] = [img_array]
                wood_name[lb_dict[v]] = [k]
            else:
                wood_data[lb_dict[v]].append(img_array)
                wood_name[lb_dict[v]].append(k)

        for i in range(7):
            wood_data[i] = tr.stack(wood_data[i])
            wood_name[i] = np.stack(wood_name[i])
            num_cls = wood_data[i].size(0)
            print(num_cls)
            num_train = int(num_cls/2)
            print(num_train)
            num_test = num_cls-num_train
            print(num_test)
            index = np.arange(num_cls)
            np.random.shuffle(index)
            train[i] = wood_data[i][index][:num_train]
            test_data.append(wood_data[i][index][num_train:])
            test_label.append(tr.LongTensor([i]).repeat(num_test))
            test_name.append(wood_name[i][index][num_train:])
        test_data = tr.cat(test_data, 0)
        test_label = tr.cat(test_label, 0)
        test_name = np.concatenate(test_name,0)
        return train,test_data,test_label, test_name




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
    # bs = prediction.size(0)
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
    args.n_way = 7
    data = data_loader(args)
    train = data.train
    test=  data.test_data
    # for i in range(7):
    #     print(train[i].size())
    train_ite = graph_train(args,train)
    # for i in range(5):
    #     print(train_ite[i].size())

    # test_ite = graph_test(args,train,test[:args.batch_size])
    # for i in range(4):
    #     print(test_ite[i].size())
    print(test.size())