import sys
import os
sys.path.append("../")

import numpy as np
import h5py
import ot
import time
from tqdm import tqdm

import gc
import sklearn
from sklearn import svm
from sklearn.manifold import MDS
# from sklearn.cluster import KMeans
import torch
from torch_kmeans import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import Axes3D


from lib.gromov_barycenter import gw_barycenters,pgw_barycenters
from lib.linear_gromov import LGW_dist,LPGW_embedding,LPGW_dist,LGW_dist,LGW_embedding,lgw_procedure,lpgw_procedure,lmpgw_procedure
from lib.opt import opt_lp,emd_lp
from lib.gromov import gromov_wasserstein, cost_matrix_d, tensor_dot_param, tensor_dot_func, gwgrad_partial1, partial_gromov_wasserstein, partial_gromov_ver1
from lib.gromov import GW_dist,MPGW_dist, PGW_dist_with_penalty

label_list=[0,2,12,15]
size1=1024
size2=768
ratio = 3 # for each complete shape, we select n incomplete shapes 
sample_size = 10
device1 = "cuda:2"
device2 = "cuda:3"

BETAS = [0.2, 0.4, 0.6, 1.0]

save_string = 'FINAL_NONORM_labels_' + '.'.join((str(l) for l in label_list)) + '_s1_' + str(size1) + '_s2_' + str(size2) + '_ratio_' + str(ratio) + '_n_' + str(sample_size)


with open('results_master/' + save_string + '.txt', 'w') as sys.stdout:
    # Adapted from Beier et al: https://github.com/Gorgotha/LGW
    def svm_cross_val(dist, X,y, gamma=10, k=10):
        dist_sq=dist
        k_folds = sklearn.model_selection.StratifiedKFold(n_splits = 10)
        k_folds.get_n_splits(X,y)

        dist = dist / dist.max()

        accs = []

        for train_index, test_index in k_folds.split(X, y):
            
            # get train and test data for this fold
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # set up SVM
            kernel_train = np.exp(-gamma * dist[X_train].T[X_train].T)
            svc = svm.SVC(kernel="precomputed")
            
            # fit SVM
            clf = svc.fit(kernel_train, y_train)

            kernel_test = np.exp(-gamma * dist_sq[X_test].T[X_train].T)
            acc = clf.score(kernel_test, y_test)
            accs.append(acc)

        return accs, np.mean(accs)

    label_to_name = {
        '0': 'airplane',
        '1': 'cabinet',
        '2': 'car',
        '3': 'chair',
        '4': 'lamp',
        '5': 'sofa',
        '6': 'table',
        '7': 'watercraft',
        '8': 'bed',
        '9': 'bench',
        '10': 'bookshelf',
        '11': 'bus',
        '12': 'guitar',
        '13': 'motorbike',
        '14': 'pistol',
        '15': 'skateboard',
    }

    with h5py.File('data/MVP_Train_CP.h5', 'r') as f:
        complete_pcds = f['complete_pcds'][:] # (2400, 2048, 3)
        partial_pcds = f['incomplete_pcds'][:] # (62400, 2048, 3)
        partial_labels = f['labels'][:] # (62400,)

    # construct the labels for complete shapes
    def partial_to_complete_indices(indices,num=26):
        n=indices.shape[0]
        complete_indices=np.zeros(n,dtype=np.int64)
        for (i,idx) in enumerate(indices):
            idx_complete= int(idx/num)
            complete_indices[i]=idx_complete
        indices_unique=np.unique(complete_indices)
        return indices_unique

    def complete_to_partial_indices(indices, num=26,size=None):
        if size is None:
            size=num
        n=indices.shape[0]
        partial_indices=[]
        for (i,idx) in enumerate(indices):
            idx_complete= idx*num+np.arange(size)
            partial_indices.append(idx_complete)
        partial_indices=np.concatenate(partial_indices)
        return partial_indices       

    complete_indices_all=[]
    for i in range(16):
        partial_indices = np.where(partial_labels==i)[0]
        print('Class %i: %i Complete Samples' % (i,partial_indices.shape[0]//26))
        complete_indices=partial_to_complete_indices(partial_indices)
        complete_indices_all.append(complete_indices)

    complete_n=(np.concatenate(complete_indices_all)).shape[0]
    print('Total: %i'%complete_n)
    print('-'*25)

    # construct complete labels
    complete_labels=np.zeros(complete_n,dtype=np.int64)
    for i,indices in enumerate(complete_indices_all):
        complete_labels[indices]=i


    data = np.load('data/MVP_train_CP.npz')

    complete_pcds = data['complete_pcds'] 
    complete_labels = data['complete_labels'] #.astype(np.float64)
    complete_pcds_sample, partial_pcds_sample, complete_labels_sample, partial_labels_sample = [],[],[],[]

    for i in label_list:
        complete_indices = np.where(complete_labels==i)[0]
        complete_indices_sample = complete_indices[0:sample_size]
        partial_indices_sample = complete_to_partial_indices(complete_indices_sample,num=26,size=ratio)
        
        complete_pcds_sample.append(complete_pcds[complete_indices_sample])
        complete_labels_sample.append(complete_labels[complete_indices_sample])
        partial_pcds_sample.append(partial_pcds[partial_indices_sample])
        partial_labels_sample.append(partial_labels[partial_indices_sample])
    complete_pcds_sample,complete_labels_sample,partial_pcds_sample,partial_labels_sample=np.concatenate(complete_pcds_sample),np.concatenate(complete_labels_sample),np.concatenate(partial_pcds_sample),np.concatenate(partial_labels_sample)



    def reduce_point_cloud(point_cloud, n_clusters, device):
        n=point_cloud.shape[1]
        if n_clusters==n:
            return point_cloud

        # Initialize the KMeans algorithm with the desired number of clusters
        kmeans = KMeans(n_clusters=n_clusters).to(device)
        data = torch.tensor(point_cloud).to(device)
        cluster_result = kmeans(data)
        centers = cluster_result.centers
        centers_cpu = centers.cpu()
        res = centers_cpu.numpy()
        del kmeans, data, centers_cpu, centers, cluster_result
        return res

    complete_pcds_sample_1=[]
    partial_pcds_sample_1=[]

    complete_pcds_sample_1 = reduce_point_cloud(complete_pcds_sample, size1, device1)
    gc.collect()
    torch.cuda.empty_cache()

    partial_pcds_samples = []
    for i in range(ratio):
        total_complete = len(label_list) * sample_size
        partial_pcds_samples += list(reduce_point_cloud(partial_pcds_sample[i*total_complete:(i+1)*total_complete], size2, device2))
        gc.collect()
        torch.cuda.empty_cache()

    pcds = list(complete_pcds_sample_1) + partial_pcds_samples
    partial_pcds_sample_1 = np.stack(partial_pcds_samples)

    labels_1=np.concatenate((complete_labels_sample,partial_labels_sample))

    print('-'*25)
    print(f"Number of Samples: {len(pcds)}")


    # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), subplot_kw={'projection': '3d'})
    # indices = [0, 0, 0, 0]
    # titles = [label_to_name[str(i)] for i in label_list]

    # elev_angle_list = [-35,-35,-35,-35]  # Elevation angle
    # azim_angle_list = [90,90,90,90]  # Azimuth angle

    # for i,(idx,title,elev_angle,azim_angle) in enumerate(zip(indices,titles,elev_angle_list,azim_angle_list)):
    #     idx1=i*sample_size+idx
    #     idx2=sample_size*len(label_list)+i*sample_size*ratio+idx
    #     for j,idx3 in enumerate([idx1,idx2]):
            
    #         ax = axes[j, i]
    #         ax.scatter(pcds[idx3][:,0], pcds[idx3][:,1],pcds[idx3][:,2],c='b', marker='o',s=10)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.set_zticks([])
    #         if j==0:
    #             ax.set_title(title,fontsize=20)
    #         ax.view_init(elev=elev_angle, azim=azim_angle)
    #         ax.set_axis_off()

    # # Add vertical text labels on the left side of the first column
    # fig.text(0.1, 0.75, 'complete', va='center', rotation='vertical', fontsize=20)
    # fig.text(0.1, 0.3 , 'incomplete', va='center', rotation='vertical', fontsize=20)

    # # Adjust layout to remove the blank space between subplots
    # plt.subplots_adjust(wspace=0., hspace=0.)  # Control space between plots

    # # Add a horizontal line between the two rows
    # fig.add_artist(plt.Line2D([0.15, .90], [0.5, 0.5], color='black', lw=2, transform=fig.transFigure))
    # for i in range(3):
    #     fig.add_artist(plt.Line2D([i/5+0.32, i/5+0.32], [0.15, 0.9], color='black', lw=2, transform=fig.transFigure))
    # plt.savefig(f"results_master/{save_string}.pdf", dpi=300, bbox_inches='tight')
    # plt.close()


    heights=[]
    size_list=np.zeros(len(pcds))
    Ms=[]
    Lambda_Max=0
    M_max=0
    for i,pcd in enumerate(pcds):
        n=pcd.shape[0]
        size_list[i]=n
        height=np.ones(n)
        heights.append(height)
        M=np.sqrt(cost_matrix_d(pcd,pcd)).astype(np.float64)
        M_max=max(M.max(),M_max)
        #M=M/M.max()
        #pcd=pcd/M.max()
        Ms.append(M)
        Lambda_Max=max(Lambda_Max,M.max())
    alpha=1/np.median(size_list)
    heights=[height*alpha for height in heights]

    Ms=[M/M_max for M in Ms]
    print("PMF Total Mass:", heights[0].sum())
    print("Dist Mat Shape:", Ms[1].shape)
    print("Number Dist Mats", len(Ms))



    # numItem=sample_size
    # idx_bary = [c*numItem for c in range(len(label_list))] #indices of shapes for barycenter computation
    # n_bary = len(idx_bary)
    # k_bary = size1
    # height_ref = np.ones(k_bary)*alpha
    # ps =[heights[s]/heights[s].sum()*height_ref.sum() for s in idx_bary]
        
    # st = time.time()   

    # Cs=[Ms[s] for s in idx_bary]
    # Lambda_list=[1.0]*len(idx_bary)
    # Cs=np.array(Cs)
    # M_ref=np.mean(Cs,0)

    # height_ref=np.ones(M_ref.shape[0])*alpha

    # M_ref = pgw_barycenters(k_bary, Cs, ps=ps, p=height_ref, lambdas=ot.unif(n_bary),Lambda_list=Lambda_list, loss_fun='square_loss',
    #         max_iter=20, tol=1e-5,stop_criterion='barycenter', verbose=False,
    #         log=False, init_C=None, random_state=0)
    # et = time.time()

    # print("Time for barycenter computation:", et-st)

    def normalize_pmf(height,total_mass=False,alpha=1.0/300):
        if total_mass==True:
            return height/height.sum()
        elif total_mass==False:
            return height/height.mean()*alpha


    # heights_normalized=[height/height.sum() for height in heights]
    # for (loss_id,loss) in enumerate(['sqrt','square']):
    #     print('loss is',loss)
        
    #     X=np.arange(len(Ms))
    #     st = time.time()
    #     if loss_id==0:
    #         lgw,(embeddings,bps)=lgw_procedure(M_ref,height_ref/height_ref.sum(),pcds,Ms,heights_normalized,loss = loss) 
    #     else:
    #         lgw,(embeddings,bps)=lgw_procedure(M_ref,height_ref/height_ref.sum(),pcds,Ms,heights_normalized,loss = loss,emb=(embeddings,bps))
    #     lgw=lgw/lgw.max()
    #     et = time.time()

    #     np.savez(f"results_master/{save_string}_LGW",dist=lgw,time=et-st)

    #     gammas = [0.01, 0.1, 0.5, 1, 2.5, 5, 10, 100]
    #     accs = [svm_cross_val(lgw/lgw.max(),X,labels_1,gamma=gamma)[1] for gamma in gammas]
    #     print(f"Best LGW Accuracy: {np.max(accs)} (Gamma = {gammas[np.argmax(accs)]})")
    #     print("Time for LGW computation:", et-st)


    # LAMBDA_MAX=1.0
    # X=np.arange(len(pcds))

    # # alpha=1/400
    # height_ref1 = normalize_pmf(height_ref,alpha=alpha)
    # heights1 = [normalize_pmf(height,alpha=alpha) for height in heights]

    # for beta in BETAS:
    #     for (loss_id,loss) in enumerate(['sqrt']):
    #         print('Loss:', loss)
    #         Lambda = beta * LAMBDA_MAX
    #         print('Lambda:',Lambda)

    #         if loss_id==0:
    #             st = time.time()
    #             lpgw_trans,lpgw_penalty, (embeddings,bps)=lpgw_procedure(M_ref,height_ref1,pcds,Ms,heights1,Lambda=Lambda,loss = loss)
    #             et = time.time()

    #         lpgw=np.sqrt(lpgw_trans+lpgw_penalty)
    #         lpgw=lpgw/lpgw.max()
            
    #         np.savez(f"results_master/{save_string}_LPGW_{str(beta)}",dist=lpgw,time=et-st)

    #         gammas = [0.01, 0.1, 0.5, 1, 2.5, 5, 10, 100]
    #         accs = [svm_cross_val(lpgw/lpgw.max(),X,labels_1,gamma=gamma)[1] for gamma in gammas]
    #         print(f"Best LPGW Accuracy: {np.max(accs)} (Gamma = {gammas[np.argmax(accs)]})")
    #         print("Time for LPGW computation:", et-st)

    X = np.arange(len(pcds))
    # N=len(Ms)
    # st = time.time()        
    # gw = np.zeros((N,N))
    # for i in tqdm(range(N)):
    #     M1 = Ms[i].copy()
    #     height1 = heights[i]
    #     for j in range(i+1, N):
    #         M2 = Ms[j].copy()
    #         height2 = heights[j]

    #         gamma = gromov_wasserstein(M1, M2, height1/height1.sum(), height2/height2.sum(), G0=None,thres=1, numItermax=100*N, tol=1e-5,log=False, verbose=False,line_search=True)
    #         gw[i, j] = GW_dist(M1, M2, gamma)

    # gw += gw.T
    # gw = np.sqrt(gw)
    # et = time.time()

    # gammas = [0.01, 0.1, 0.5, 1, 2.5, 5, 10, 100]
    # accs = [svm_cross_val(gw/gw.max(),X,labels_1,gamma=gamma)[1] for gamma in gammas]
    # print(f"Best GW Accuracy: {np.max(accs)} (Gamma = {gammas[np.argmax(accs)]})")
    # print("GW computation: " + str(np.round(et-st,2)) + "s")


    # np.savez(f'results_master/{save_string}_GW',dist=gw,time=et-st)

    print("Starting PGW Computation", flush=True)

    N=len(Ms)
    pgw = np.zeros((N,N))
    Lambda_MAX=1.0
    # heights1 = [normalize_pmf(height,alpha=alpha) for height in heights]
    heights1 = heights

    for beta in BETAS:
        st = time.time()        
        Lambda = beta * Lambda_MAX
        print('PGW Lambda:', Lambda)
        for i in tqdm(range(N)):
            M1 = Ms[i]
            height1 = heights1[i]
        
            for j in range(i+1, N):
                M2 = Ms[j]
                height2 = heights1[j]
        
                gamma = partial_gromov_ver1(M1, M2, height1, height2, G0=None, Lambda=Lambda, thres=1, numItermax=100*N, tol=1e-5,log=False, verbose=False,line_search=True)
        
                pgw_trans,pgw_penalty=PGW_dist_with_penalty(M1,M2,gamma, height1, height2, Lambda)
                pgw[i, j]=pgw_trans+pgw_penalty 
        
        pgw[pgw < 0] = 0
        pgw += pgw.T
        pgw = np.sqrt(pgw)
        et = time.time()
        np.savez(f'results_master/{save_string}_PGW_{beta}',dist=pgw,time=et-st)

        gammas = [0.01, 0.1, 0.5, 1, 2.5, 5, 10, 100]
        accs = [svm_cross_val(pgw/pgw.max(),X,labels_1,gamma=gamma)[1] for gamma in gammas]
        print(f"Best PGW Accuracy: {np.max(accs)} (Gamma = {gammas[np.argmax(accs)]})", flush=True)
        print("PGW computation: " + str(np.round(et-st,2)) + "s")