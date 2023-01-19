import numpy as np

def rank(dist_results, dist_correct, query_index):
        return np.sum(dist_results[query_index,:] <= dist_results[query_index, np.nanargmin(dist_correct, axis=1)[query_index]].reshape(query_index.shape[0],1),axis=1)

def reciprocal_rank(dist_results, dist_correct, query_index):
    return 1./np.sum(dist_results[query_index,:] <= dist_results[query_index, np.nanargmin(dist_correct, axis=1)[query_index]].reshape(query_index.shape[0],1),axis=1)

def mean_reciprocal_rank(dist_results, dist_correct, query_index):
    return np.mean(reciprocal_rank(dist_results, dist_correct, query_index))


def reciprocal_rank_smallest(df_results, df_corrects, k):
    df_corrects_small =  df_corrects.nsmallest(k+1, df_corrects.columns)
    sml_idx = df_corrects_small.index.values
    df_results_small = df_results.loc[sml_idx,1:]
    return reciprocal_rank(df_results_small.values, df_corrects_small.values, np.arange(0,len(df_corrects_small)))


def mean_reciprocal_rank_smallest(df_results, df_corrects, k):
    return np.mean(reciprocal_rank_smallest(df_results, df_corrects, k+1))

def reciprocal_rank_filter(df_results, df_corrects, filter_correct, max_cor, max_res):
    df_corrects_small =  df_corrects[df_corrects <= filter_correct]
    df_results_small = df_results[df_corrects <= filter_correct]
    df_results_small[~df_corrects_small.notna()] = max_res
    df_corrects_small[~df_corrects_small.notna()] = max_cor
    return reciprocal_rank_filter_(df_results_small.values, df_corrects_small.values, np.arange(0,len(df_corrects_small)), max_res)

def reciprocal_rank_filter_(dist_results, dist_correct, query_index, max_res):
    
    rr = np.sum(np.multiply(
      dist_results[query_index,:] <= dist_results[query_index, np.nanargmin(dist_correct, axis=1)[query_index]].reshape(query_index.shape[0],1),
      (dist_results[query_index,:] < max_res)),axis=1) 

    idx = rr==0
    rr = 1./rr

    rr[idx] = 0
    
    return rr

def mean_reciprocal_rank_filter(df_results, df_corrects, filter_correct, max_cor, max_res):
    return np.mean(reciprocal_rank_filter(df_results, df_corrects, filter_correct, max_cor, max_res))