import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torchvision.models as model


class distance_func(nn.Module):
    ### output a similarity matrix
    def __init__(self, args, in_dimen, ratio = [2, 2, 1, 1], dropout = False):
        super(distance_func, self).__init__()
        self.args = args
        nf_adj = args.nf_adj

        self.Module_list = []
        for i in range(len(ratio)):
            if i == 0:
                self.Module_list.append(nn.Conv2d(in_dimen, nf_adj*ratio[0], 1))
                if dropout:
                    self.Module_list.append(nn.Dropout(0.6))
            else:
                self.Module_list.append(nn.Conv2d(nf_adj*ratio[i-1], nf_adj*ratio[i], 1))
            self.Module_list.append(nn.BatchNorm2d(nf_adj*ratio[i]))
            self.Module_list.append(nn.LeakyReLU())
        self.Module_list.append(nn.Conv2d(nf_adj*ratio[-1], 1, 1))
        self.Module_list = nn.ModuleList(self.Module_list)

    def forward(self, input):
        ### input size is (bs, N, feature_dimen)
        similarity = self.subtraction(input)
        for l in self.Module_list:
            similarity = l(similarity)
        similarity = tr.transpose(similarity, 1, 3)

        similarity = similarity.squeeze(-1)

        return similarity


    def subtraction(self, input):
        ### input size is (bs, N, feature_dimens)   where N is n_way*n_shot + 1 which needs to be predicted

        A_x = input.unsqueeze(2) # A_x size is (bs, N, 1, feature_dimen)
        A_y = tr.transpose(A_x, 1, 2) # A_y size is (bs, 1, N, feature_dimen)
        subtraction = tr.abs(A_x - A_y) # A_update size is (bs, N, N, feature_dimen)

        subtraction = tr.transpose(subtraction, 1, 3) # A_update size is (bs, feature_dimen, N, N)

        return subtraction


class GCN(nn.Module):
    ### input is [image_embedding, label]
    def __init__(self, args, input_dimens, output_dimens):
        ###
        super(GCN, self).__init__()
        self.args = args
        self.feature_dimen = input_dimens

        self.output  =nn.Linear(self.feature_dimen, output_dimens)

    def forward(self, input_, adj):
        ### adj size is (bs, N, N)
        ### input_ size is (bs, N, feature_dimens)
        u = tr.bmm(adj, input_)
        ### u size is (bs, N, features)

        h = self.output(u)


        return h



class Adj_update(nn.Module):
    def __init__(self, args, input_dimens,ratio = [2,2,1,1]):
        super(Adj_update, self).__init__()

        self.distance_matrix = distance_func(args, input_dimens,ratio=ratio)

    def forward(self, adj_before, input):

        W_init = tr.eye(input.size(1)).unsqueeze(0).repeat(input.size(0), 1, 1)
        if tr.cuda.is_available:
            W_init = W_init.cuda()
        # time0 = time.time()
        distance_matrix = self.distance_matrix(input)
        # print('constructing a graph need time {}'.format(time.time() - time0))
        distance_matrix = distance_matrix - W_init*1e8
        distance_matrix = F.softmax(distance_matrix, 2)

        adj_after = adj_before - distance_matrix

        return adj_after

class Anchor(nn.Module):
    def __init__(self, args, indimen, unkpre = False):
        super(Anchor, self).__init__()
        self.args = args
        self.n_way = args.n_way
        self.n_shot = args.n_shot
        self.bs = args.batch_size
        self.emb = args.embedding_size
        dimen = indimen
        self.dimen = dimen
        self.unkpre = unkpre
        # ratio = [1,2,1]
        self.Anchor_producing = nn.Sequential(
                nn.Linear(dimen, dimen),
                nn.BatchNorm1d(dimen),
                nn.LeakyReLU()
            )

        # self.Anchor_producing = nn.Sequential(
        #     nn.Linear(dimen, dimen),
        #     nn.BatchNorm1d(dimen),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(dimen, dimen),
        #     nn.BatchNorm1d(dimen),
        #     nn.LeakyReLU(),
        # )


        self.Anchor_cat = nn.Sequential(
                nn.Linear(self.n_way*self.n_shot*dimen, 2*dimen),
                nn.BatchNorm1d(2*dimen),
                nn.LeakyReLU(),

                nn.Linear(2*dimen, dimen),
                nn.BatchNorm1d(dimen),
                nn.LeakyReLU()
            )


        self.Anchor_using = nn.Sequential(
                # nn.Linear(dimen * 2, dimen * 2),
                # nn.BatchNorm1d(dimen * 2),
                # nn.LeakyReLU(),

                nn.Linear(2*dimen, dimen),
                nn.BatchNorm1d(dimen)
                # nn.LeakyReLU(),
                #
                # nn.Linear(dimen, 1),
                # nn.Sigmoid()
            )

        self.ratio = nn.Sequential(

            nn.Linear(self.bs*self.n_way*dimen, dimen),
            nn.LeakyReLU(),

            nn.Linear(self.dimen,1),
            nn.Sigmoid()
        )

        # self.ratio = nn.Sequential(
        #
        #     nn.Linear(self.bs * self.n_way * dimen, 1),
        #     # nn.LeakyReLU(),
        #     #
        #     # nn.Linear(self.dimen, 1),
        #     nn.Sigmoid()
        # )

        self.ratio_anchor = nn.Sequential(

            nn.Linear(self.bs*dimen, 1),
            nn.Sigmoid()
        )

        if unkpre:
            ratio = [1, 1, 1]
            sim = []
            for i in range(len(ratio) - 1):
                sim.append(nn.Linear(dimen * ratio[i], dimen * ratio[i + 1], 1))
                sim.append(nn.BatchNorm1d(dimen * ratio[i + 1]))
                sim.append(nn.LeakyReLU())
            sim.append(nn.Linear(dimen * ratio[-1], 1, 1))
            sim.append(nn.Sigmoid())
            self.sim = nn.Sequential(*sim)


    def forward(self, x):

        x_unl = x[:,0,:].unsqueeze(1)  ### size is (bs, 1, feature_size)
        xu_feat = x_unl[:,:,:-self.n_way]

        x_l = x[:,1:,:]
        x_label = x_l[:,:,-self.n_way:]### size is (bs, n_way * n_shot, n_way)
        x_feat = x_l[:,:,:-self.n_way]###  size is (bs, n_way * n_shot, embedding_size)

        ### anchor producing mean
        x_anchor = tr.bmm(tr.transpose(x_label,1,2),x_feat)/self.n_shot ### size is (bs, n_way, embedding_size)
        # x_anchor_size = x_anchor.size()
        # x_anchor = x_anchor.view(-1,x_anchor_size[-1])
        # x_anchor = self.Anchor_producing(x_anchor)
        # x_anchor = x_anchor.view(x_anchor_size[0],x_anchor_size[1],-1)
        # # x_anchor_ = x_anchor
        x_anchor = tr.bmm(x_label, x_anchor) ### size is (bs, n_way*n_shot, embedding_size)

        ### anchor producing mean
        # x_label_ = tr.transpose(x_label,1,2).unsqueeze(-1).repeat(1,1,1,self.dimen)
        # x_feat_ = x_feat.unsqueeze(1).repeat(1,self.n_way,1,1)
        # x_anchor = x_label_*x_feat_
        # x_anchor_size = x_anchor.size() ### size is (bs, n_way, N, feat)
        # x_anchor = x_anchor.view(x_anchor_size[0]*x_anchor_size[1],x_anchor_size[2]*x_anchor_size[3])
        # x_anchor = self.Anchor_cat(x_anchor)
        # x_anchor = x_anchor.view(*x_anchor_size[:2],-1)
        # x_anchor = tr.bmm(x_label, x_anchor)  ### size is (bs, n_way*n_shot, embedding_size)


        ### cat nonlinear
        # x_anchorfeat = tr.cat((x_anchor,x_feat),2)
        # x_size = x_anchorfeat.size()
        # x_anchorfeat = x_anchorfeat.view(-1,x_size[-1])
        # x_output = self.Anchor_using(x_anchorfeat)
        # x_output = x_output.view(x_size[0],x_size[1],-1)


        ### plus nonlinear
        # x_anchorfeat = (x_anchor+x_feat)/2
        # x_size = x_anchorfeat.size()
        # x_anchorfeat = x_anchorfeat.view(-1,x_size[-1])
        # x_output = self.Anchor_using(x_anchorfeat)
        # x_output = x_output.view(x_size[0],x_size[1],-1)



        ###plus
        # x_output = (x_anchor+x_feat)/2


        ### ratio plus
        # x_output = (x_anchor+x_feat)/2
        # x_output = x_anchor_
        # x_output = x_output.view(-1)
        # ratio = self.ratio(x_output)
        # print(ratio.size())
        x_output = 0.5*x_anchor+0.5*x_feat

        ### ratio anchor plus
        # x_output = (x_anchor+x_feat)/2
        # x_output = tr.transpose(tr.bmm(tr.transpose(x_label,1,2),x_output)/self.n_shot,0,1)
        # x_output_size = x_output.size()
        # x_output = x_output.contiguous()
        # x_output = x_output.view(x_output_size[0],-1)
        # ratio = self.ratio_anchor(x_output)
        # ratio = tr.bmm(x_label,ratio.unsqueeze(0).repeat(self.bs,1,self.dimen))
        # x_output = ratio*x_anchor+(1-ratio)*x_feat

        ### x_output size is (bs, n_way*n_shot, feature)


        out = tr.cat((x_output, x_label),2)
        out = tr.cat((x_unl,out),1)
        return out



class gnn(nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        self.args = args
        self.nf_gc = args.nf_gc
        self.input_dimens = args.embedding_size + args.n_way
        self.num_layer = args.num_layer_gc
        self.Adj_update = []
        self.Gnn_update = []
        self.anchor = []
        self.n_way = args.n_way
        self.n_anchor = 2

        for i in range(self.num_layer - 1):
            if i == 0:
                self.anchor.append(Anchor(args, args.embedding_size,unkpre=False))

                self.Adj_update.append(Adj_update(args, self.input_dimens))
                self.Gnn_update.append(GCN(args, self.input_dimens, self.nf_gc))


            else:
                self.anchor.append(Anchor(args, args.embedding_size+self.nf_gc*i))

                self.Adj_update.append(Adj_update(args, self.input_dimens + self.nf_gc * (i )))
                self.Gnn_update.append(GCN(args, self.input_dimens + self.nf_gc * i, self.nf_gc))

        self.anchor = nn.ModuleList(self.anchor)
        self.Adj_update = nn.ModuleList(self.Adj_update)
        self.Gnn_update = nn.ModuleList(self.Gnn_update)

        self.anchor_final = Anchor(args, args.embedding_size+self.nf_gc*(self.num_layer - 1))
        self.Adj_update_final = Adj_update(args, self.input_dimens + self.nf_gc*(self.num_layer - 1))
        self.Graph_convolution_final = GCN(args, self.input_dimens + self.nf_gc*(self.num_layer - 1), self.n_way)
        print('Adjacency Module is {}'.format(self.Adj_update))
        print('GNN Module is {}'.format(self.Gnn_update))
        print('Anchor Module is {}'.format(self.anchor))
    def forward(self, input, training = True):

        ### input size is (bs, N, num_feature) where num_feature is cat(embedding_size, n_way )

        #

        A_init = tr.eye(input.size(1)).unsqueeze(0).repeat(input.size(0), 1, 1)
        if tr.cuda.is_available:
            A_init = A_init.cuda()
        # A_update = A_init
        # Gc_update = self.Anchor(input)
        Gc_update =input


        for i in range(self.num_layer - 1):
            # if i < self.n_anchor:
            Gc_update = self.anchor[i](Gc_update)

            A_update = self.Adj_update[i](A_init, Gc_update)

            Gc_update_new = F.leaky_relu(self.Gnn_update[i](Gc_update, A_update)) ### size is (bs, N, num_feature)

            Gc_update = tr.cat([Gc_update_new, Gc_update], 2)

        Gc_update = self.anchor_final(Gc_update)
        A_final = self.Adj_update_final(A_init, Gc_update)
        Gn_final = self.Graph_convolution_final(Gc_update, A_final)

        if training:

            return Gn_final[:,0,:]

        else:

            if self.args.visual:
                return Gn_final[:,0,:], Gc_update[:,0,:]
            else:
                return Gn_final[:,0,:]




class Embedding(nn.Module):
    def __init__(self, args, freez = True, use_pretrain = True):
        super(Embedding, self).__init__()

        ### Alexnet fix layer
        # self.net = model.alexnet(pretrained=use_pretrain)
        # self.freez(self.net, freez)
        # indimen = self.net.classifier[6].in_features
        # self.net.classifier[6] = nn.Linear(indimen,args.embedding_size)

        ### alexnet fine tune
        vgg11 = model.vgg11(pretrained=use_pretrain)
        self.alexnet_extractor = vgg11.features
        self.alexnet_extractor_low = self.alexnet_extractor[:16] ###freeze
        self.alexnet_extractor_high = self.alexnet_extractor[16:] ### fine tune
        self.alexnet_avgpool = vgg11.avgpool
        self.alexnet_classifier = vgg11.classifier
        self.freez(self.alexnet_extractor_low, freez)
        indimen = self.alexnet_classifier[6].in_features

        self.alexnet_classifier[6] = nn.Sequential(  ### fine tune in top 5 layers and tune in the last layer

            nn.Linear(indimen, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, args.embedding_size)
        )


    def forward(self,x):
        if x.size(1) == 1:
            x = x.repeat(1,3,1,1)

        ### Alexnet
        features_low = self.alexnet_extractor_low(x)
        features_high = self.alexnet_avgpool(self.alexnet_extractor_high(features_low))
        features_flatten = tr.flatten(features_high,1)
        # features = self.features_extra(self.alexnet_classifier(features_flatten))
        features = self.alexnet_classifier(features_flatten)
        # features = self.net(x)

        return features

    def freez(self, model, freez = True):
        if freez == True:
            for para in model.parameters():
                para.requires_grad = False




if __name__ == '__main__':
    from argument import args
    # import time
    args = args()
    # N = args.n_shot*args.n_way+1
    # args.embedding_size = 32
    # args.nf_adj = 32
    # args.nf_gc = 48
    # for _ in range(10):
    #
    #     comp_time_v = tr.randn(2,N,args.embedding_size)
    #     time0 = time.time()
    #     construct = distance_func(args,args.embedding_size)
    #
    #     construct(comp_time_v)
    #     print(time.time() - time0)

    x = tr.randn(20,1,64,64)
    test = Embedding(args)
    b = test(x)
    print(b.size())