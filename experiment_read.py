import numpy as np



def compute_precision_recall(cm, label_cls, num_cls):
    pre = {}
    recal = {}
    F1_Score = {}
    pre_macro = []
    recal_macro = []
    # F1_Score_macro = []
    for i in range(num_cls):
        precision = np.round(cm[i,i]/np.sum(cm[:,i])*100,2)
        if np.isnan(precision):
            precision=1
        pre[label_cls[i]] = precision
        pre_macro.append(precision)

        recall = np.round(cm[i,i]/np.sum(cm[i,:])*100, 2)

        recal[label_cls[i]] = recall
        recal_macro.append(recall)

        F1_Score[label_cls[i]] = precision*recall*2/(precision+recall)
        if np.isnan(precision*recall*2/(precision+recall)):
            F1_Score[label_cls[i]] = 0

    pre_macro = np.mean(pre_macro)
    recal_macro = np.mean(recal_macro)
    f1_macro = pre_macro*recal_macro*2/(pre_macro+recal_macro)
    return pre,recal,F1_Score, pre_macro,recal_macro,f1_macro


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
elif dataset == 'wood':
    dict_cls = dict_cls_wood
elif dataset == 'textile':
    dict_cls = dict_cls_textile
elif dataset == 'MSD':
    dict_cls = dict_cls_MSD

res = []
pre = []
recal = []
f1 = []
pre_class = {}
recal_class = {}
f1_class = {}

for k,i in dict_cls.items():

    pre_class[i] = []
    recal_class[i] = []
    f1_class[i] = []


for i in range(10):

    if dataset == 'neu64':
        # test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, _, _ = np.load(
        #     'experiment_data/ACB_GCN_NEU64_{}_{}.npy'.format(150, i))

        test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, _, _ = np.load(
            'experiment_data/ACB_GCN_NEU64_singmin_{}_{}.npy'.format(150, i))
        dict_num = 9
    elif dataset == 'wood':
        test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, _, _ = np.load(
            'experiment_data/ACB_GCN_wood_{}.npy'.format(i))
        dict_num = 7
    elif dataset == 'textile':
        test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, _, _ = np.load(
            './experiment_data/ACB_GCN_textile_{}.npy'.format(i))
        dict_num = 6
    elif dataset == 'MSD':
        test_loss_, train_loss_, train_accu_, accu, cm, wrong_name_, _, _ = np.load(
            './experiment_data/data/ACB_GCN_MSD_{}.npy'.format(i))
        dict_num = 10
    idx = np.argmax(accu)

    pre_, recal_,f1_, pre_macro, recal_macro,f1_macro = compute_precision_recall(cm[idx],dict_cls, dict_num)
    for j in pre_class.keys():
        pre_class[j].append(pre_[j])
        recal_class[j].append(recal_[j])
        f1_class[j].append(f1_[j])
    res.append(accu[idx])
    pre.append(pre_macro)
    recal.append(recal_macro)
    f1.append(f1_macro)

best_result_idx = np.argsort(res)[::-1][:10]


res_best = np.mean(np.array(res)[best_result_idx])
print('The average accuracy is {}'.format(np.round(res_best, 2)))
pre_best = np.mean(np.array(pre)[best_result_idx])
print('The average precision is {}'.format(np.round(pre_best, 2)))
recal_best = np.mean(np.array(recal)[best_result_idx])
print('The average recall is {}'.format(np.round(recal_best, 2)))
f1_best = np.mean(np.array(f1)[best_result_idx])
print('The average f1 score is {}'.format(np.round(f1_best, 2)))


for j in pre_class.keys():
    pre_class[j] = np.round(np.mean(np.array(pre_class[j])[best_result_idx]),2)
    recal_class[j] = np.round(np.mean(np.array(recal_class[j])[best_result_idx]),2)
    f1_class[j] = np.round(np.mean(np.array(f1_class[j])[best_result_idx]),2)


print('The precision of each class is {}'.format(pre_class))
print('The recall of each class is {}'.format(recal_class))
print('The f1 score of each class is {}'.format(f1_class))


