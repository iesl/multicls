import faiss
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from sklearn.manifold import MDS
import pandas as pd
#import json
import ast
import os
import pickle

class Neighbor_finder:
    def __init__(self, view):
        self.view=view
        self.index = faiss.index_factory(768, "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(self.view)
        self.index.add(view)
        
    def get_neighbor(self,text_id, n, text):
        
        q = np.expand_dims(self.view[text_id],axis=0)
        D, I = self.index.search(q, n)
        self.print_text(I,text, skip=True)
        return D, I
    
    def get_neighor_by_vector(self, q, n):
        q = np.expand_dims(q,axis=0)
        D, I = self.index.search(q, n)
        #self.print_text(I, skip)
        return D, I
        
    def print_text(self, I, text, skip=False):
        counter=1
        t=I[0].tolist()
        if skip:
            t.pop(0)
        
        for i in t:
            print("Neighbor "+str(counter)+" :"+ text[i])
            counter=counter+1
 
class Finder_by_Method:
    def __init__(self, method, text, num_facets):
        self.method=method
        self.raw_text=text
        self.view_arr = []
        for i in range(1, num_facets+1):
            self.view_arr.append(np.load('../vis_emb/{}/view_{}.npy'.format(method,i))) 
        #self.view_1=np.load('../vis_emb/{}/view_1.npy'.format(method))
        #self.view_2=np.load('../vis_emb/{}/view_2.npy'.format(method))
        #self.view_3=np.load('../vis_emb/{}/view_3.npy'.format(method))
        self.create_finder()
    
    def create_finder(self):
        num_facets = len(self.view_arr)
        self.finder_view_arr = []
        for i in range(num_facets):
            self.finder_view_arr.append( Neighbor_finder(self.view_arr[i]) )
        #self.finder_view_1=Neighbor_finder(self.view_1)
        #self.finder_view_2=Neighbor_finder(self.view_2)
        #self.finder_view_3=Neighbor_finder(self.view_3)
    
    def get_view_among_all(self, finder_view, choose_id, n):
        all_D=[]
        all_I=[]

        D, I = finder_view.get_neighor_by_vector(self.view_1[choose_id],n=n[0]+1)
        all_D.append(D[0][-1])
        all_I.append(I[0][-1])
        print('q_1:',self.raw_text[I[0][-1]])
    #     for i in I[0]:
    #         print('q_1:',raw_text[i])

        D, I = finder_view.get_neighor_by_vector(self.view_2[choose_id],n=n[1]+1)
        all_D.append(D[0][-1])
        all_I.append(I[0][-1])
        print('q_2:',self.raw_text[I[0][-1]])
    #     for i in I[0]:
    #         print('q_2:', raw_text[i])

        D, I = finder_view.get_neighor_by_vector(self.view_3[choose_id],n=n[2]+1)
        all_D.append(D[0][-1])
        all_I.append(I[0][-1]) 
        print('q_3:',self.raw_text[I[0][-1]])
    #     for i in I[0]:
    #         print('q_3:', raw_text[i])
        print('Choose facet:', np.argmax(all_D)+1)
        print('Max:', self.raw_text[all_I[np.argmax(all_D)]])
        
    
    def vis(self, index, model, plot_edges=False):
        num_facets = len(self.view_arr)
        #X=np.concatenate([self.view_1[index],self.view_2[index], self.view_3[index]])
        X = np.concatenate([self.view_arr[i][index] for i in range(num_facets)])
        if model=='tsne':
            X_embedded = TSNE(n_components=2).fit_transform(X)
        else:
            X_embedded = MDS(n_components=2).fit_transform(X)
        X_embedded = X_embedded.reshape(num_facets,-1,2)
        
        #c_map={1:'r', 2:'b',3:'g'}
        #for i in c_map:
            #plt.scatter(X_embedded[i-1][:,0], X_embedded[i-1][:,1], color=c_map[i], label="facet "+str(i))
        for i in range(num_facets):
            plt.scatter(X_embedded[i][:,0], X_embedded[i][:,1], label="facet "+str(i+1))
        
        if plot_edges:
            for j in range(index.shape[0]):
                X_emb_avg = 0
                for i in range(num_facets):
                    X_emb_avg += X_embedded[i][j,:]
                X_emb_avg /= num_facets
                for i in range(num_facets):
                    plt.plot([ X_embedded[i][j,0], X_emb_avg[0]], [ X_embedded[i][j,1], X_emb_avg[1]])
                

        plt.legend(loc="upper left")
        plt.title(self.method)
        plt.show()
    

class Finder_by_CSV(Finder_by_Method):
    def __init__(self, file_path):
        self.method = file_path
        cached_features_file = file_path + '_cache'
        overwrite_cache = True
        #overwrite_cache = False
        if os.path.exists(cached_features_file) and not overwrite_cache:
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.raw_text, self.view_arr, self.predict, self.true_label, self.imp_arr_list, self.imp_text = pickle.load(handle)
        else:
            print("Creating features from dataset file at %s", cached_features_file)

            df = pd.read_csv(file_path, sep="\t")
            #print(df['sentence_2'][0])
            self.raw_text = ['Prediction: ' + str(df['prediction'][j]) + ' +++ Label: ' + str(df['true_label'][j]) + ' +++ ' + df['sentence_1'][j] + ' +++ ' + str(df['sentence_2'][j]) for j in range(len(df.index)) ]
            self.imp_text = [df['sentence_1'][j] + ' [SEP] ' + str(df['sentence_2'][j]) for j in range(len(df.index)) ]
            num_facets = len(ast.literal_eval(df['facet_emb'][0]))
            #print(num_facets)
            view_arr_list = [[] for i in range(num_facets)]
            self.imp_arr_list = [[] for i in range(num_facets)]
            #print(df['facet_emb'][0])
            #print(ast.literal_eval(df['facet_emb'][0]))

            for j in range(len(df.index)):
                facet_list = ast.literal_eval(df['facet_emb'][j])
                importance_list = ast.literal_eval(df['facet_importance'][j])
                #print(len(facet_list))
                for i in range(num_facets):
                    #print( json.loads(df['facet_emb'][j]) )
                    #facet_list = json.loads(df['facet_emb'][j])
                    view_arr_list[i].append(facet_list[i])
                    self.imp_arr_list[i].append(importance_list[i])
            
            self.view_arr = []
            #self.imp_arr_list = []
            for i in range(num_facets):
                self.view_arr.append(np.array(view_arr_list[i], dtype=np.float32))
                #print(imp_arr_list[i])
                #self.imp_arr_list.append( np.array(imp_arr_list[i], dtype=np.float32) )
                #print(self.view_arr[i].shape)
            #self.predict = df['prediction'].tolist()
            try:
                self.predict = df['prediction'].to_numpy()
            except:
                self.predict = df['prediction'].tolist()
            self.true_label = df['true_label'].to_numpy()
            
            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump( [self.raw_text, self.view_arr, self.predict, self.true_label, self.imp_arr_list, self.imp_text], handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.create_finder()

    def find_strange_idx(self):
        def var_of_embedding(inputs, dim):
            #(batch, facet, embedding)
            #inputs_norm = inputs / np.linalg.norm(inputs. dim=-1, keepdim=True)
            inputs = inputs / inputs.sum(axis = -1, keepdims=True)
            pred_mean = inputs.mean(axis=dim, keepdims=True)
            loss_set_div = np.linalg.norm(inputs - pred_mean, axis=-1)
            return loss_set_div
    
        def compute_imp_var(imp_arr_list, idx):
            imp_arr = np.zeros( (num_facets, len(imp_arr_list[0][idx]) ) )
            for i in range(num_facets):
                imp_arr[i,:] = np.array(imp_arr_list[i][idx] )
            var_idx = var_of_embedding(imp_arr, dim=0).mean()
            return var_idx
            #num_facets
        K = 10
        num_facets = len(self.view_arr)
        I_arr = []
        for i in range(num_facets):
            D, I = self.finder_view_arr[i].index.search(self.finder_view_arr[i].view, K)
            I_arr.append(I)
        weird_idx_arr = []
        for j in range( len(self.raw_text) ):
            var_j = compute_imp_var(self.imp_arr_list, j)
            overall_pred = self.predict[j]
            inconsist_arr = []
            for i in range(num_facets):
                inconsistent_count = 0
                for k in range(K):
                    neighbor_pred = self.predict[I_arr[i][j][k]]
                    #print(neighbor_pred)
                    #print(overall_pred)
                    if neighbor_pred != overall_pred:
                        inconsistent_count += 1
                inconsist_arr.append( inconsistent_count / float(K) )
            if max(inconsist_arr) > 0.5:
                weird_idx_arr.append( str([j, inconsist_arr, var_j, len( self.imp_arr_list[i][j] ) ]) )
                #self.finder_view_arr[i].get_neighbor(j, k, finder.raw_text)
        print("weird ratio:", len(weird_idx_arr) / len(self.raw_text))
        print('Weird index:', '\n'.join(weird_idx_arr))
        print('\n')



    def vis(self, index, model, plot_edges=False):
        #X=np.concatenate([self.view_1[index],self.view_2[index], self.view_3[index]])
        num_facets = len(self.view_arr)
        X = np.concatenate([self.view_arr[i][index] for i in range(num_facets)])
        if model=='tsne':
            X_embedded = TSNE(n_components=2).fit_transform(X)
        else:
            X_embedded = MDS(n_components=2).fit_transform(X)
        X_embedded = X_embedded.reshape(len(self.view_arr),-1,2)
        
        c_map={1:'r', 2:'b',3:'g'}
        #for i in c_map:
            #plt.scatter(X_embedded[i-1][:,0], X_embedded[i-1][:,1], color=c_map[i], label="facet "+str(i))

        load_file_name = os.path.basename(self.method)
        if True:
        #if load_file_name == 'rte_val.tsv' or load_file_name == 'cola_val.tsv' or load_file_name == 'mnli_val.tsv':
            if False:
                label_mask_0 = self.true_label[index] == 0
                label_mask_1 = self.true_label[index] == 1
                label_mask_2 = self.true_label[index] == 2
            elif 'rte' in self.method:
                label_mask_0 = self.predict[index] == "not_entailment"
                label_mask_1 = self.predict[index] == "entailment"
                label_mask_2 = self.predict[index] == 2
            else:
                label_mask_0 = self.predict[index] == 0
                label_mask_1 = self.predict[index] == 1
                label_mask_2 = self.predict[index] == 2
            for i in range(num_facets):
                plt.scatter(X_embedded[i][label_mask_0,0], X_embedded[i][label_mask_0,1], marker='x', label="facet "+str(i+1))
                plt.scatter(X_embedded[i][label_mask_1,0], X_embedded[i][label_mask_1,1], marker='o', label="facet "+str(i+1))
                #plt.scatter(X_embedded[i][label_mask_2,0], X_embedded[i][label_mask_2,1], marker='^', label="facet "+str(i+1))

            #X_sum = X_embedded[0]
            X_sum = 0
            for i in range(num_facets):
                X_sum = X_sum + X_embedded[i]
            X_sum /= num_facets
            plt.scatter(X_sum[label_mask_0,0], X_sum[label_mask_0,1], marker='x', label="facet sum")
            plt.scatter(X_sum[label_mask_1,0], X_sum[label_mask_1,1], marker='o', label="facet sum")
            #plt.scatter(X_sum[label_mask_2,0], X_sum[label_mask_2,1], marker='^', label="facet sum")


        #print(num_facets)
        if plot_edges:
            for i in range(num_facets):
            #for i in range(1):
                print(i)
                #plt.plot([100, 100], [-100, -100])
                for j in range(index.shape[0]):
                    #print(X_embedded[i][j,:],  X_sum[j,:])
                    #plt.plot([ X_embedded[i][j,0], X_sum[j,0]], [ X_embedded[i][j,1], X_sum[j,1]], 'o-', color=c_map[i+1], linewidth=2)
                    plt.plot([ X_sum[j,0], X_embedded[i][j,0]], [ X_sum[j,1],  X_embedded[i][j,1]], color=c_map[i+1])
                    if i ==0:
                        #plt.text(X_sum[j,0], X_sum[j,1], str(index[j]))
                        plt.text(X_embedded[i][j,0], X_embedded[i][j,1], str(index[j]))
            #for i in range(1):
            #    print(i)
            #    #plt.plot([100, 100], [-100, -100])
            #    for j in range(index.shape[0]):
            #        #print(X_embedded[i][j,:],  X_sum[j,:])
            #        plt.plot([ X_embedded[i][j,0], X_sum[j,0]], [ X_embedded[i][j,1], X_sum[j,1]], color=c_map[i+1], linewidth=2)
        #plt.plot([100, 100], [-100, -100])


        #plt.legend()
        #plt.legend(loc="upper left")
        plt.title(self.method)
        plt.show()
