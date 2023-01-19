from math import sin, cos, sqrt, atan2, radians
import numpy as np

def warping_path(D):
    
    n = D.shape[0] - 1
    m = D.shape[1] - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            min_val = (0, m - 1)
        elif m == 0:
            min_val = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                min_val = (n-1, m-1)
            elif val == D[n-1, m]:
                min_val = (n-1, m)
            else:
                min_val = (n, m-1)
        P.append(min_val)
        (n, m) = min_val
    P.reverse()
    return np.array(P)

def euclidean_distance(x1, y1, x2, y2):
    
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(x1)
    lon1 = radians(y1)
    lat2 = radians(x2)
    lon2 = radians(y2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def dtw_distance(tid1, tid2, df): 

    df1 = df.loc[df['trajectory_id'] == tid1].reset_index() 
    df2 = df.loc[df['trajectory_id'] == tid2].reset_index()
    len_tj1 = len(df1) 
    len_tj2 = len(df2)

    DTW = np.zeros((len_tj1, len_tj2)) 

    
    for j in range(1, len_tj2):
        DTW[0,j] = euclidean_distance(df1['lat'][0], df1['lon'][0], df2['lat'][j], df2['lon'][j]) + DTW[0,j-1]

   
    for i in range(1, len_tj1):
        DTW[i,0] = euclidean_distance(df2['lat'][0], df2['lon'][0], df1['lat'][i], df1['lon'][i]) + DTW[i-1,0]


    for i in range(1, len_tj1):
        for j in range(1, len_tj2):
            cost = euclidean_distance(df1['lat'][i], df1['lon'][i], df2['lat'][j], df2['lon'][j]) 
            DTW[i,j] = cost + min(DTW[i-1,j],DTW[i,j-1], DTW[i-1,j-1]) 

    wp = warping_path(DTW) 
    DTW_valor = 0
    
    for i in range(len(wp)):
        DTW_valor += DTW[wp[i][0], wp[i][1]] 

    DTW_valor = DTW_valor/len(wp)

    return DTW_valor

def edit_distance(traj1, traj2): 
    
    if len(traj1) > len(traj2):
        difference = len(traj1) - len(traj2)
        for i in range(len(traj2)):
            if traj2[i] != traj1[i]:
                difference += 1

    elif len(traj2) > len(traj1):
        difference = len(traj2) - len(traj1)
        for i in range(len(traj1)):
            if traj1[i] != traj2[i]:
                difference += 1

    else:
        difference = 0
        for i in range(len(traj1)):
            if traj1[i] != traj2[i]:
                difference += 1


    return difference