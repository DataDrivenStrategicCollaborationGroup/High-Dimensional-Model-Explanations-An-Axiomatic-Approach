import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import binom

class BanzhafModel(object):
    def __init__(self, model, baseline_values):
        self.model = model
        self.baseline_values = baseline_values 
    def predict_proba(self, x, input, factor=1):
        input_vector = self.baseline_values.copy().repeat(len(input),axis=0)
        input_vector[input] = x.repeat(len(input),axis=0)[input]
        if hasattr(self.model,'predict_proba'):
            prediction = factor*self.model.predict_proba(input_vector)[:,1] 
        else:
            prediction = factor*self.model.predict(input_vector)
        return    prediction.sum(axis=0)
    def predict_proba_sparse(self, input_vector, factor=1):
        # Sparse vectors have always baseline 0
        prediction = self.model.predict_proba(input_vector)[:,1][None].T
        prediction = np.multiply(factor,prediction)
        return    prediction.sum(axis=0)
    def predict_sparse(self, input_vector):
        # Sparse vectors have always baseline 0
        return    self.model.predict(input_vector).sum(axis=0)

def banzhaf_sparse(x, banzhaf_model, number_samples=100):
    n_features = x.nnz
    banzhaf = np.zeros(x.shape[1])
    indices = x.indices
    for feature in indices:
        row_ind = np.arange(number_samples).repeat(n_features).reshape(number_samples,n_features).flatten()
        col_ind = indices.repeat(number_samples).reshape(number_samples,len(indices)).flatten()
        input = csr_matrix(((1 == np.random.choice(2,size=[number_samples,n_features])).flatten(),
                            (row_ind, col_ind)),shape=(number_samples,x.shape[1]))
        input[:,feature] = True
        banzhaf[feature] += banzhaf_model.predict_proba_sparse(input)
        input[:,feature] = False
        banzhaf[feature] -= banzhaf_model.predict_proba_sparse(input)
    return banzhaf   / number_samples  


def shapley_sparse(x,shapley_model,number_samples=100):
    n_features = x.nnz
    shapley = np.zeros(x.shape[1])
    indices = x.indices
    for feature in indices:
        row_ind = np.arange(number_samples).repeat(n_features).reshape(number_samples,n_features).flatten()
        col_ind = indices.repeat(number_samples).reshape(number_samples,len(indices)).flatten()
        input = csr_matrix(((1 == np.random.choice(2,size=[number_samples,n_features])).flatten(),
                        (row_ind, col_ind)),shape=(number_samples,x.shape[1]))
        factor = 1/binom(n_features,input.sum(axis=1))
        input[:,feature] = True
        shapley[feature] += shapley_model.predict_proba_sparse(input, factor=factor)
        input[:,feature] = False
        shapley[feature] -= shapley_model.predict_proba_sparse(input, factor=factor)
    return shapley   / number_samples

def binary_banzhaf_sparse(x,banzhaf_model,number_samples=100):
    n_features = x.nnz
    banzhaf = np.zeros([x.shape[1],x.shape[1]])
    indices = x.indices
    for i_num, feature_i in enumerate(indices):
        for j_num in range(i_num+1,len(indices)):
            feature_j = indices[j_num]
            row_ind = np.arange(number_samples).repeat(n_features).reshape(number_samples,n_features).flatten()
            col_ind = indices.repeat(number_samples).reshape(number_samples,len(indices)).flatten()
            data = (1 == np.random.choice(2,size=[number_samples,n_features])).flatten()
            input = csr_matrix((data, (row_ind, col_ind)),shape=(number_samples,x.shape[1]))
            factor = 1/binom(n_features,input.sum(axis=1))
            input[:,feature_i] = False
            input[:,feature_j] = False
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba_sparse(input, factor=factor)
            input[:,feature_i] = False
            input[:,feature_j] = True
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba_sparse(input, factor=factor)
            input[:,feature_i] = True
            input[:,feature_j] = False
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba_sparse(input, factor=factor)
            input[:,feature_i] = True
            input[:,feature_j] = True
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba_sparse(input, factor=factor)
    return banzhaf / number_samples 

def binary_shapley_sparse(x,banzhaf_model,number_samples=100):
    n_features = x.nnz
    banzhaf = np.zeros([x.shape[1],x.shape[1]])
    indices = x.indices
    for i_num, feature_i in enumerate(indices):
        for j_num in range(i_num+1,len(indices)):
            feature_j = indices[j_num]
            row_ind = np.arange(number_samples).repeat(n_features).reshape(number_samples,n_features).flatten()
            col_ind = indices.repeat(number_samples).reshape(number_samples,len(indices)).flatten()
            data = (1 == np.random.choice(2,size=[number_samples,n_features])).flatten()
            input = csr_matrix((data, (row_ind, col_ind)),shape=(number_samples,x.shape[1]))
            input[:,feature_i] = False
            input[:,feature_j] = False
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba_sparse(input)
            input[:,feature_i] = False
            input[:,feature_j] = True
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba_sparse(input)
            input[:,feature_i] = True
            input[:,feature_j] = False
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba_sparse(input)
            input[:,feature_i] = True
            input[:,feature_j] = True
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba_sparse(input)
    return banzhaf / number_samples


class OneHotMapping(object):
    def __init__(self, one_hot_columns, cont_columns, cat_columns):
        self.output_len = len(one_hot_columns)
        self.cont_len = len(cont_columns)
        self.cat_len = len(cat_columns)
        self.cont_pos = np.asarray([one_hot_columns.index(c) for c in cont_columns])
        self.cat_pos = np.zeros(self.cat_len).astype(int)
        self.category_len =  np.zeros(self.cat_len).astype(int)
        for i in range(self.cat_len):
            boolean_cat = np.asarray([cat_columns[i] in c for c in  one_hot_columns])
            self.cat_pos[i] = int(boolean_cat.argmax())
            self.category_len[i] = int(boolean_cat.sum())
    def map_to(self, input):
        one_hot_columns = np.zeros([len(input),self.output_len ])
        for i in range(self.cont_len):
            one_hot_columns[:,self.cont_pos[i]] = input[:,i]
        for i in range(self.cat_len):
            one_hot_columns[:,self.cat_pos[i]:self.cat_pos[i]+self.category_len[i]] =\
                input[:,self.cont_len+i][:,None].repeat(self.category_len[i],axis=1)
        return one_hot_columns == 1
    
def one_hot_banzhaf(x,banzhaf_model, cont_columns,cat_columns, one_hot_columns, number_samples=100):
    n_features = len(cont_columns) + len(cat_columns)
    mapping = OneHotMapping(one_hot_columns,cont_columns,cat_columns)
    banzhaf = np.zeros(n_features)
    for feature in range(n_features):
        input = (1 == np.random.choice(2,size=[number_samples,n_features]))
        input[:,feature] = True
        one_hot_columns = mapping.map_to(input)
        banzhaf[feature] += banzhaf_model.predict_proba(x, one_hot_columns)
        input[:,feature] = False
        one_hot_columns = mapping.map_to(input)
        banzhaf[feature] -= banzhaf_model.predict_proba(x,one_hot_columns)
    return banzhaf   / number_samples   

def one_hot_shapley(x, banzhaf_model, cont_columns,cat_columns, one_hot_columns, number_samples=100):
    n_features = len(cont_columns) + len(cat_columns)
    mapping = OneHotMapping(one_hot_columns,cont_columns,cat_columns)
    banzhaf = np.zeros(n_features)
    for feature in range(n_features):
        input = (1 == np.random.choice(2,size=[number_samples,n_features]))
        factor = 1/binom(n_features,input.sum(axis=1))
        input[:,feature] = True
        one_hot_columns = mapping.map_to(input)
        banzhaf[feature] += banzhaf_model.predict_proba(x, one_hot_columns, factor=factor)
        input[:,feature] = False
        one_hot_columns = mapping.map_to(input)
        banzhaf[feature] -= banzhaf_model.predict_proba(x,one_hot_columns, factor=factor)
    return banzhaf   / number_samples


def one_hot_binary_banzhaf(x,banzhaf_model,cont_columns,cat_columns, one_hot_columns, number_samples=100):
    n_features = len(cont_columns) + len(cat_columns)
    mapping = OneHotMapping(one_hot_columns,cont_columns,cat_columns)
    banzhaf = np.zeros([n_features,n_features])
    for feature_i in range(n_features):
        for feature_j in range(feature_i+1,n_features):
            input = (1 == np.random.choice(2,size=[number_samples,n_features]))
            input[:,feature_i] = False
            input[:,feature_j] = False
            one_hot_columns = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x, one_hot_columns)
            input[:,feature_i] = False
            input[:,feature_j] = True
            one_hot_columns = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x,one_hot_columns)
            input[:,feature_i] = True
            input[:,feature_j] = False
            one_hot_columns = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x,one_hot_columns)
            input[:,feature_i] = True
            input[:,feature_j] = True
            one_hot_columns = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x,one_hot_columns)
    return banzhaf / number_samples 

def one_hot_binary_qii(x,banzhaf_model,cont_columns,cat_columns, one_hot_columns):
    n_features = len(cont_columns) + len(cat_columns)
    mapping = OneHotMapping(one_hot_columns,cont_columns,cat_columns)
    banzhaf = np.zeros([n_features,n_features])
    for feature_i in range(n_features):
        for feature_j in range(feature_i+1,n_features):
            input = (1 == np.ones([1,n_features]))
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x, mapped_input)
            input[:,feature_i] = False
            input[:,feature_j] = False
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x, mapped_input)
    return banzhaf  
            
def one_hot_qii(x,banzhaf_model,cont_columns,cat_columns,  one_hot_columns):
    n_features = len(cont_columns) + len(cat_columns)
    mapping = OneHotMapping(one_hot_columns,cont_columns,cat_columns)
    banzhaf = np.zeros(n_features)
    for feature_i in range(n_features):
        input = (1 == np.ones([1,n_features]))
        mapped_input = mapping.map_to(input)
        banzhaf[feature_i] += banzhaf_model.predict_proba(x, mapped_input)
        input[:,feature_i] = False
        mapped_input = mapping.map_to(input)
        banzhaf[feature_i] -= banzhaf_model.predict_proba(x, mapped_input)
    return banzhaf  

def one_hot_binary_shapley(x,banzhaf_model, cont_columns, cat_columns,  one_hot_columns, number_samples=100):
    n_features = len(cont_columns) + len(cat_columns)
    mapping = OneHotMapping(one_hot_columns,cont_columns,cat_columns)
    banzhaf = np.zeros([n_features,n_features])
    for feature_i in range(n_features):
        for feature_j in range(feature_i+1,n_features):
            input = (1 == np.random.choice(2,size=[number_samples,n_features]))
            factor = 1/binom(n_features,input.sum(axis=1))
            input[:,feature_i] = False
            input[:,feature_j] = False
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x, mapped_input, factor=factor)
            input[:,feature_i] = False
            input[:,feature_j] = True
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x,mapped_input, factor=factor)
            input[:,feature_i] = True
            input[:,feature_j] = False
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x,mapped_input, factor=factor)
            input[:,feature_i] = True
            input[:,feature_j] = True
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x,mapped_input, factor=factor)
    return banzhaf / number_samples 


class OneHotMergeMapping(object):
    def __init__(self, one_hot_columns, cont_columns,cat_columns, merged_features):
        self.output_len = len(one_hot_columns)
        self.cont_len = len(cont_columns)
        self.cat_len = len(cat_columns)
        self.cont_pos = np.asarray([one_hot_columns.index(c) for c in cont_columns])
        self.cat_pos = np.zeros(self.cat_len).astype(int)
        self.category_len =  np.zeros(self.cat_len).astype(int)
        self.merged_features = merged_features
        for i in range(self.cat_len):
            boolean_cat = np.asarray([cat_columns[i] in c for c in  one_hot_columns])
            self.cat_pos[i] = int(boolean_cat.argmax())
            self.category_len[i] = int(boolean_cat.sum())
    def map_to(self, input):
        output = np.zeros([len(input),self.output_len ])
        counter = 0
        for i in range(self.cont_len):
            if i == self.merged_features[1]:
                continue
            output[:,self.cont_pos[i]] = input[:,counter]
            if i == self.merged_features[0]:
                output[:,self.cont_pos[self.merged_features[1]]] = input[:,counter]
            counter += 1
        for i in range(self.cat_len):
            output[:,self.cat_pos[i]:self.cat_pos[i]+self.category_len[i]] = input[:,counter][:,None].repeat(self.category_len[i],axis=1)
            counter += 1
        return output == 1

def one_hot_banzhaf_merge(x,banzhaf_model,cont_columns,cat_columns, one_hot_columns, merged_features, number_samples=100):
    n_features = len(cont_columns) + len(cat_columns) - 1 
    mapping = OneHotMergeMapping(one_hot_columns,cont_columns,cat_columns,merged_features)
    banzhaf = np.zeros(n_features)
    for feature in range(n_features):
        input = (1 == np.random.choice(2,size=[number_samples,n_features]))
        input[:,feature] = True
        mapped_input = mapping.map_to(input)
        banzhaf[feature] += banzhaf_model.predict_proba(x, mapped_input)
        input[:,feature] = False
        mapped_input = mapping.map_to(input)
        banzhaf[feature] -= banzhaf_model.predict_proba(x,mapped_input)
    return banzhaf   / number_samples   

def one_hot_shapley_merge(x,banzhaf_model,cont_columns,cat_columns, one_hot_columns, merged_features, number_samples=100):
    n_features = len(cont_columns) + len(cat_columns) - 1 
    mapping = OneHotMergeMapping(one_hot_columns,cont_columns,cat_columns,merged_features)
    banzhaf = np.zeros(n_features)
    for feature in range(n_features):
        input = (1 == np.random.choice(2,size=[number_samples,n_features]))
        factor = 1/binom(n_features,input.sum(axis=1))
        input[:,feature] = True
        mapped_input = mapping.map_to(input)
        banzhaf[feature] += banzhaf_model.predict_proba(x, mapped_input, factor=factor)
        input[:,feature] = False
        mapped_input = mapping.map_to(input)
        banzhaf[feature] -= banzhaf_model.predict_proba(x,mapped_input, factor=factor)
    return banzhaf   / number_samples   

def one_hot_binary_qii_merge(x,banzhaf_model,cont_columns,cat_columns, one_hot_columns, merged_features):
    n_features = len(cont_columns) + len(cat_columns) - 1 
    mapping = OneHotMergeMapping(one_hot_columns,cont_columns,cat_columns,merged_features)
    banzhaf = np.zeros([n_features,n_features])
    for feature_i in range(n_features):
        for feature_j in range(feature_i+1,n_features):
            input = (1 == np.ones([1,n_features]))
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x, mapped_input)
            input[:,feature_i] = False
            input[:,feature_j] = False
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x, mapped_input)
    return banzhaf  
            
def one_hot_qii_merge(x,banzhaf_model,cont_columns,cat_columns, one_hot_columns, merged_features):
    n_features = len(cont_columns) + len(cat_columns) - 1 
    mapping = OneHotMergeMapping(one_hot_columns,cont_columns,cat_columns,merged_features)
    banzhaf = np.zeros(n_features)
    for feature_i in range(n_features):
        input = (1 == np.ones([1,n_features]))
        mapped_input = mapping.map_to(input)
        banzhaf[feature_i] += banzhaf_model.predict_proba(x, mapped_input)
        input[:,feature_i] = False
        mapped_input = mapping.map_to(input)
        banzhaf[feature_i] -= banzhaf_model.predict_proba(x, mapped_input)
    return banzhaf  

def one_hot_binary_shapley_merge(x,banzhaf_model,cont_columns,cat_columns, one_hot_columns, merged_features, number_samples=100):
    n_features = len(cont_columns) + len(cat_columns) - 1 
    mapping = OneHotMergeMapping(one_hot_columns,cont_columns,cat_columns,merged_features)
    banzhaf = np.zeros([n_features,n_features])
    for feature_i in range(n_features):
        for feature_j in range(feature_i+1,n_features):
            input = (1 == np.random.choice(2,size=[number_samples,n_features]))
            factor = 1/binom(n_features,input.sum(axis=1))
            input[:,feature_i] = False
            input[:,feature_j] = False
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x, mapped_input, factor=factor)
            input[:,feature_i] = False
            input[:,feature_j] = True
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x,mapped_input, factor=factor)
            input[:,feature_i] = True
            input[:,feature_j] = False
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x,mapped_input, factor=factor)
            input[:,feature_i] = True
            input[:,feature_j] = True
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x,mapped_input, factor=factor)
    return banzhaf / number_samples     
    
def one_hot_binary_banzhaf_merge(x,banzhaf_model,cont_columns,cat_columns, one_hot_columns, merged_features, number_samples=100):
    n_features = len(cont_columns) + len(cat_columns) - 1 
    mapping = OneHotMergeMapping(one_hot_columns,cont_columns,cat_columns,merged_features)
    banzhaf = np.zeros([n_features,n_features])
    for feature_i in range(n_features):
        for feature_j in range(feature_i+1,n_features):
            input = (1 == np.random.choice(2,size=[number_samples,n_features]))
            input[:,feature_i] = False
            input[:,feature_j] = False
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x, mapped_input)
            input[:,feature_i] = False
            input[:,feature_j] = True
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x,mapped_input)
            input[:,feature_i] = True
            input[:,feature_j] = False
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] -= banzhaf_model.predict_proba(x,mapped_input)
            input[:,feature_i] = True
            input[:,feature_j] = True
            mapped_input = mapping.map_to(input)
            banzhaf[feature_i,feature_j] += banzhaf_model.predict_proba(x,mapped_input)
    return banzhaf / number_samples 
