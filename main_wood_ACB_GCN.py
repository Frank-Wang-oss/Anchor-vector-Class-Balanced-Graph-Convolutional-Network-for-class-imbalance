import torch as tr
import numpy as np
import torch.optim as opt
from data_loader_wood import data_loader, graph_train, graph_test
import ACB_GCN
import torch.nn as nn
import time
import random
import os
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F



class main():
    def __init__(self, args):
        self.args = args
        data = data_loader(args)
        self.train = data.train
        self.test = self.trans_cuda(data.test)
        self.tlabel = self.trans_cuda(data.tlabel)
        self.lr = args.lr
        self.dec_lr = 1000
        self.lr_thr = 1e-4

        self.embed_module = ACB_GCN.Embedding(args)
        self.GNN_module = ACB_GCN.gnn(args)

        if tr.cuda.is_available():
            self.embed_module = self.embed_module.cuda()
            self.GNN_module = self.GNN_module.cuda()



        feature_tune_id = list(map(id, self.embed_module.alexnet_classifier[6].parameters()))
        feature_finetune = filter(lambda p:id(p) not in feature_tune_id, self.embed_module.alexnet_classifier.parameters())

        params = [
            {'params': feature_finetune, 'lr': self.lr * 0.001},
            {'params': self.embed_module.alexnet_extractor_high.parameters(), 'lr': self.lr * 0.001},
            # {'params': self.embed_module.features_extra.parameters()},
            {'params': self.embed_module.alexnet_classifier[6].parameters()}
        ]
        self.opt_embed = opt.Adam(params, lr = self.lr, weight_decay=1e-6)

        # self.opt_embed = opt.Adam(self.embed_module.parameters(), lr=self.lr, weight_decay=1e-6)
        self.opt_gnn = opt.Adam(self.GNN_module.parameters(), lr = self.lr, weight_decay=1e-6)
        # self.loss = nn.CrossEntropyLoss(weight=tr.Tensor(weights).cuda())
        self.loss = nn.CrossEntropyLoss()
        self.add_count = 0
        self.bs = args.batch_size

    def Train_batch(self, iteration):
        self.embed_module.train()
        self.GNN_module.train()

        data = graph_train(self.args, self.train)
        if tr.cuda.is_available():
            data_cuda = [data_.cuda() for data_ in data]
            data = data_cuda

        self.opt_embed.zero_grad()
        self.opt_gnn.zero_grad()

        xi, xi_label, xs, xs_label, xl_onehot = data

        xi_embed = self.embed_module(xi)


        xs_embed_ = [self.embed_module(xs[:,i,:,:,:]) for i in range(xs.size(1))]
        xs_embed = tr.stack(xs_embed_, 1)

        x_features = tr.cat((xi_embed.unsqueeze(1),xs_embed), 1)

        uniform_pad = tr.Tensor(xl_onehot.size(0), 1, xl_onehot.size(2)).fill_(1.0 / xl_onehot.size(2)).cuda()
        # uniform_pad = self.initial_label(xl_onehot)

        x_label = tr.cat([uniform_pad, xl_onehot], 1)

        nodes_feature = tr.cat([x_features, x_label], 2)

        out = self.GNN_module(nodes_feature)

        loss = self.loss(out, xi_label)

        loss.backward()

        self.opt_embed.step()
        self.opt_gnn.step()

        self.adjust_learning_rate(optimizers=[self.opt_embed,self.opt_gnn], lr = self.lr, iter=iteration)

        return loss.item()


    def prediction(self, unlabel_input, visual = False):
        self.embed_module.eval()
        self.GNN_module.eval()

        data = graph_test(self.args, self.train, unlabel_input)
        if tr.cuda.is_available():
            data_cuda = [data_.cuda() for data_ in data]
            data = data_cuda

        [xi, xs, xs_label, xl_onehot] = data

        xi_embed = self.embed_module(xi)

        xs_embed_ = [self.embed_module(xs[:, i, :, :, :]) for i in range(xs.size(1))]
        xs_embed = tr.stack(xs_embed_, 1)

        x_features = tr.cat((xi_embed.unsqueeze(1), xs_embed), 1)

        uniform_pad = tr.Tensor(xl_onehot.size(0), 1, xl_onehot.size(2)).fill_(1.0 / xl_onehot.size(2)).cuda()
        # uniform_pad = self.initial_label(xl_onehot)

        x_label = tr.cat([uniform_pad, xl_onehot], 1)

        nodes_feature = tr.cat([x_features, x_label], 2)

        if visual and self.args.visual:
            out,visual = self.GNN_module(nodes_feature, training = False)
            return out.detach(),visual.detach()

        else:

            out = self.GNN_module(nodes_feature)
        # out = tr.argmax(out, 1)

            return out.detach()

    def eval_train(self):
        self.embed_module.eval()
        self.GNN_module.eval()

        correct = 0
        total = 0
        sample_eval = self.args.sample_eval
        iteration = int(sample_eval/self.args.batch_size)
        for i in range(iteration):

            data = graph_train(self.args, self.train)
            if tr.cuda.is_available():
                data_cuda = [data_.cuda() for data_ in data]
                data = data_cuda

            [xi, xi_label, xs, xs_label, xl_onehot] = data

            xi_embed = self.embed_module(xi)

            xs_embed_ = [self.embed_module(xs[:, i, :, :, :]) for i in range(xs.size(1))]
            xs_embed = tr.stack(xs_embed_, 1)

            x_features = tr.cat((xi_embed.unsqueeze(1), xs_embed), 1)

            uniform_pad = tr.Tensor(xl_onehot.size(0), 1, xl_onehot.size(2)).fill_(1.0 / xl_onehot.size(2)).cuda()
            # uniform_pad = self.initial_label(xl_onehot)

            x_label = tr.cat([uniform_pad, xl_onehot], 1)

            nodes_feature = tr.cat([x_features, x_label], 2)

            out = self.GNN_module(nodes_feature)
            out = tr.argmax(out, 1)
            for j in range(out.size(0)):
                total = total + 1
                if out[j] == xi_label[j]:
                    correct = correct + 1

        accu = (correct/total)*100


        return accu, correct, total


    def train_imb(self):
        train_loss_ = []
        test_loss_ = []
        train_accu_ = []
        test_accu_ = []
        confusion_m_ = []
        visual_vector_ = []

        time0 = time.time()
        wrong_name_ = []
        count = 0
        # num = int((self.train_unlabeled.size(0))/self.args.num_cycle)
        loss_comp = 0
        loss_count = 0
        for i in range(self.args.iteration):
            loss = self.Train_batch(i)
            loss_comp = loss_comp + loss
            loss_count = loss_count + 1

            if i% self.args.interval == 0:
                self.add_count = self.add_count + 1

                loss = loss_comp/loss_count
                loss_comp = 0
                loss_count = 0
                train_accu, correct_train, total_train = self.eval_train()

                test_accu,total,correct,loss_test, confusion_m,visual_vector = self.test_eval()
                time_interval = time.time() - time0

                print('------------The {}th iteration--------------'.format(i))
                print('count is {}'.format(count))
                print('Training loss is {}'.format(loss))
                print('Testing loss is {}'.format(loss_test))
                print('Training accuracy is {}, correct/total is {}/{}'.format(train_accu,correct_train,total_train))
                print('Testing accuracy is {}, correct/total is {}/{}'.format(test_accu,correct, total))
                print('Cost time is {}'.format(time_interval))
                print('This is {}'.format(self.args.save_name))
                visual_vector_.append(visual_vector)
                test_loss_.append(loss_test)
                train_loss_.append(loss)
                train_accu_.append(train_accu)
                test_accu_.append(test_accu)
                confusion_m_.append(confusion_m)
                # wrong_name_.append(wrong_name)
                time0 = time.time()
        final_test_accu = self.test_eval()
        print('------------Final testing accuracy is {}'.format(final_test_accu))
        np.save('./experiment_data/{}.npy'.format(self.args.save_name),[test_loss_,train_loss_,train_accu_,test_accu_,confusion_m_,wrong_name_, visual_vector_,self.tlabel.cpu().numpy()])
        # tr.save(self.embed_module.state_dict(),'./checkpoint/embed_para_{}.pth'.format(self.args.save_name))
        # tr.save(self.GNN_module.state_dict(), './checkpoint/GNN_para_{}.pth'.format(self.args.save_name))

    def test_eval(self):
        self.embed_module.eval()
        self.GNN_module.eval()
        real = []
        prediction = []
        loss = 0
        total = 0
        correct = 0
        num = self.test.size(0)
        bs = self.args.batch_size
        iteration = int(num/bs)
        wrong_name = []
        visual_vector_ = []
        remain = num-(iteration*bs)
        for i in range(iteration+1):
            if i < iteration:
                prediction_data = self.test[i*bs:(i+1)*bs]
                label_data = self.tlabel[i*bs:(i+1)*bs]
            if i==iteration:
                prediction_data = self.test[-remain:]
                label_data = self.tlabel[-remain:]
                bs = args.batch_size
                args.batch_size = remain
            # label_data_name = self.test_name[count*bs:(count+1)*bs]
            if self.args.visual:
                out, visual_vector = self.prediction(prediction_data, visual = True)
                visual_vector = visual_vector.cpu().numpy()
                visual_vector_.append(visual_vector)
            else:
                out = self.prediction(prediction_data)

            loss_i = self.loss(out,label_data)
            loss = loss + loss_i.item()


            bs_out = self.trans_cuda(tr.zeros(self.args.batch_size, self.args.n_way))

            for _ in range(self.args.pre_num):
                out = F.log_softmax(self.prediction(prediction_data), 1)  ### size is (18,6)
                bs_out = bs_out + out
            args.batch_size = bs
            # print(args.batch_size)
            bs_out = tr.argmax(bs_out, 1)
            for j in range(bs_out.size(0)):
                total = total + 1
                if bs_out[j] == label_data[j]:
                    correct = correct + 1
                # elif bs_out[j] != label_data[j]:
                #     wrong_name.append(label_data_name[j])
            real.append(label_data.cpu().numpy())
            prediction.append(bs_out.cpu().numpy())
        real = np.concatenate(real,0)
        prediction = np.concatenate(prediction, 0)
        # wrong_name = np.stack(wrong_name,0)
        loss = loss/iteration
        accu = (correct/total)*100
        confusion_m = confusion_matrix(real, prediction)
        # visual_vector_ = np.concatenate(visual_vector_,1)
        # print(visual_vector_[0].shape)
        return accu, total, correct, loss, confusion_m,visual_vector_

    def trans_cuda(self, input):
        if tr.cuda.is_available():
            input = input.cuda()
        return input

    def mode(self, list, dim):
        out_mode = []
        stack_list = tr.stack(list, dim)
        for i in range(stack_list.size(0)):
            sample_i = stack_list[i]
            mode = tr.bincount(sample_i)
            mode = tr.argmax(mode)
            out_mode.append(mode)
        out_mode = tr.stack(out_mode, 0)

        return out_mode


    def adjust_learning_rate(self, optimizers, lr, iter):
        # if lr>self.lr_thr:
        #     new_lr = lr * (0.5 ** (int(iter / self.dec_lr)))
        # else:
        #     new_lr = lr
        new_lr = lr * (0.5 ** (int(iter / self.dec_lr)))
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def initial_label(self, xl_onehot):
        uniform_pad = tr.Tensor(1, xl_onehot.size(2))
        num_data = [len(self.train[i]) for i in range(len(self.train))]
        print(num_data)
        num_data = sum(num_data)

        for i in range(uniform_pad.size(-1)):
            uniform_pad[:,i] = (len(self.train[i])/num_data)
        uniform_pad = (1-uniform_pad/tr.max(uniform_pad))
        uniform_pad = uniform_pad.repeat(xl_onehot.size(0),1,1).cuda()

        return uniform_pad
    def class_weight(self, x):
        y = []
        max_n = max(x)
        for i in x:
            y.append(max_n/i)
        # print(y)
        return y
if __name__ == '__main__':
    from argument import args
    import random
    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tr.manual_seed(seed)
        tr.cuda.manual_seed(seed)
        tr.backends.cudnn.deterministic = True

    args = args()

    args.n_way = 7
    args.data_seed = 0
    args.sample_eval = 300
    # for i in range(20):
    #     args.lr = 1e-4
    #     args.save_name = 'Anchor_wood_{}'.format(i)
    #     Train = main(args)
    #     Train.train_imb()
    args.lr = 1e-4
    args.visual = True
    args.save_name = 'Anchor_wood_visulize_0'
    Train = main(args)
    Train.train_imb()