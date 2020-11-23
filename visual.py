import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

#####draw graph

def compute_distance(x1,x2,y, num_cls):
    def compute_cls_dist(cls1, cls2):
        ###cls1 size is (120,2),cls2 size is (120,2)
        x1_ = np.expand_dims(cls1, 0)
        x2_ = np.expand_dims(cls2, 1)
        dist = np.sqrt(np.sum(np.square(x1_ - x2_), 2))

        dist = np.mean(dist)

        return dist
    intra = 0
    inter = 0
    x = np.stack((x1,x2),1)
    data_cls = {}
    for i in range(num_cls):
        idx_i = np.where(y == i)
        data_cls[i] = x[idx_i]
    ###compute intra:
    dist_mat = np.zeros((num_cls,num_cls))
    for i in range(num_cls):
        for j in range(num_cls):
            dist_mat[i][j] = compute_cls_dist(data_cls[i],data_cls[j])

    for i in range(num_cls):
        intra = intra + dist_mat[i][i]

    for i in range(num_cls):
        for j in range(num_cls):
            if i!=j:
                inter = inter + dist_mat[i][j]
    intra = intra/num_cls
    inter = inter/(num_cls**2-1)

    # print(dist_mat)
    # print(intra)
    # print(inter)
    return intra,inter

def test_visual(x, y, label, num_cls, name=None):
    fig = plt.figure()
    sub = fig.add_subplot(1, 1, 1)
    color = ['r', 'b', 'y', 'c', 'm', 'g', 'k','lime','indigo','wheat']
    # print(x.shape)
    tsne = TSNE(n_components=2)
    reduce_dimension = tsne.fit_transform(x)
    x_ = (reduce_dimension[:, 0] - np.min(reduce_dimension[:, 0])) \
         / (np.max(reduce_dimension[:, 0]) - np.min(reduce_dimension[:, 0]))
    y_ = (reduce_dimension[:, 1] - np.min(reduce_dimension[:, 1])) \
         / (np.max(reduce_dimension[:, 1]) - np.min(reduce_dimension[:, 1]))
    # print(reduce_dimension.shape)
    for i in range(num_cls):
        idx = np.where(y == i)

        sub.scatter(x_[idx], y_[idx], s=30, c=color[i], alpha=0.8, edgecolors='w', linewidths=1, label=label[i])

    intra,inter = compute_distance(x_,y_,y, num_cls)

    sub.legend(loc=1)
    sub.set_title(name)
    if name is not None:
        plt.savefig('./{}.png'.format(name))
    plt.show()
    plt.close()
    return intra,inter



dict_cls_wood = {0: 'dry_knot', 1: 'encased_knot', 2: 'horn_knot', 3: 'decayed_knot', 4: 'leaf_knot', 5: 'edge_knot',6:'sound_knot'}

dict_cls_neu64 = { 0: 'cr', 1: 'gg', 2: 'in', 3: 'pa', 4: 'ps', 5: 'rp', 6:'rs',7:'sc',8:'sp'}

dict_cls_textile = {0: 'color', 1:'cut', 2: 'good', 3: 'hole', 4:'metal_contamination', 5:'thread'}

dict_cls_MSD = {
    0: 'crease', 1: 'crescent_gap', 2: 'inclusion', 3: 'oil_spot', 4: 'punching_hole',
    5: 'rolled_pit', 6: 'silk_spot', 7: 'waist folding', 8: 'water_spot', 9: 'welding_line'
}

dataset = 'MSD'

if dataset == 'neu64':
    dict_cls = dict_cls_neu64
    test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, visual_vector_, label = np.load('./experiment_data/ACB_GCN_NEU64_visual.npy')
    dict_num = 9

elif dataset == 'wood':
    dict_cls = dict_cls_wood
    test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, visual_vector_, label = np.load('./experiment_data/ACB_GCN_wood_visual.npy')
    dict_num = 7

elif dataset == 'textile':
    dict_cls = dict_cls_textile
    test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, visual_vector_, label = np.load('./experiment_data/data/ACB_GCN_textile_visual.npy')
    dict_num = 6

elif dataset == 'MSD':
    dict_cls = dict_cls_MSD
    test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, visual_vector_, label = np.load('./experiment_data/data/ACB_GCN_MSD_visual.npy')
    dict_num = 10





i = 4
intra,inter = test_visual(x = np.concatenate(visual_vector_[i],0),y = label,label = dict_cls,num_cls = dict_num, name = None)

print(intra)
print(inter)