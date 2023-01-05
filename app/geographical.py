from keras_preprocessing.sequence import pad_sequences

def padding_trajs(trajetorias_seq, max_seq_length):
      
        X = pad_sequences(trajetorias_seq,maxlen = max_seq_length,dtype = "int32",padding = "pre",truncating = "pre",value = 0)
        X_ = X[:,:max_seq_length-1]
        Y = X[:,max_seq_length-1]
        X_ = X_[Y!=0,:]  
        Y_ = Y[Y!=0]    
        return X_, Y_