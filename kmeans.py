'''This module stores the clustering functions to be used for the training process
'''
import cv2
import numpy as np
from itertools import combinations

# Clustered Online Cumulative K-Means (CLOCK) 
def onl_kmeans(data, fnames, max_clusters, max_range, min_size = 2):
    '''Cluster the given data and pick the biggest cluster

    Params
    ---
    - data: A list of images' coordinates
    - fnames: Corresponding list of the image names
    - max_clusters: Max number of cluster to export. Putting 0 or -1 will get all available clusters
    - max_range: Max distance from centroid to be considered part of a cluster
    - min_size: The clustering will run until at least one cluster reach the specified minimum size

    Returns
    ---
    - cluster_elements: Point coordinates of the chosen cluster
    - cluster_filenames: Image filename of the chosen cluster
    - cluster_central: The centroid of the chosen cluster
    '''
    cluster_centrals = None
    cluster_elems = []
    cluster_filename = []
    cluster_count = []
    for i,coord in enumerate(data):
        # Adding the first point as the first cluster central
        if cluster_centrals is None:
            cluster_centrals = np.array([coord])
            cluster_elems.append([coord])
            cluster_filename.append([fnames[i]])
            cluster_count.append(1)
            continue
        # Get distance from point to each cluster
        distances = np.sum((cluster_centrals - coord)**2, axis=1)**0.5
        nearest = np.argmin(distances)
        # If point is far away from clusters, it's on its own cluster
        if distances[nearest] > max_range:
            # Stop when max number of clusters reached and have a big enough cluster
            if cluster_centrals.shape[0] >= max_clusters \
                and np.max(cluster_count) >= min_size \
                and max_clusters > 0 : break
            cluster_centrals = np.append(cluster_centrals,[coord], axis=0)
            cluster_elems.append([coord])
            cluster_filename.append([fnames[i]])
            cluster_count.append(1)
        # If not, it belongs to cluster with nearest centeal. Update that one
        else:
            cluster_centrals[nearest] = (cluster_centrals[nearest] 
                                       * cluster_count[nearest] 
                                       + coord) / (cluster_count[nearest]+1)
            cluster_elems[nearest].append(coord)
            cluster_filename[nearest].append(fnames[i])
            cluster_count[nearest] += 1
    # Return the coordinates, filenames, and center of the largest cluster
    biggest_cluster = np.argmax(cluster_count)
    return cluster_elems[biggest_cluster], \
           cluster_filename[biggest_cluster], \
           cluster_centrals[biggest_cluster]

# Similar Histograms Online Clustered K-Means (SHOCK) 
def hist_onl_kmeans(data, hist, fnames, max_clusters, max_range, min_size = 1, take_best_hist = False):
    '''Cluster the given data and pick the best cluster, using voting strategy

    Params
    ---
    - data: A list of images' coordinates
    - hist: The test image's histogram
    - fnames: Corresponding list of the image names
    - max_clusters: Max number of cluster to export. Putting 0 or -1 will get all available clusters
    - max_range: Max distance from centroid to be considered part of a cluster
    - min_size: The clustering will run until at least one cluster reach the specified minimum size
    - take_best_hist: If True, will choose the cluster that best match the color

    Returns
    ---
    - cluster_elements: Point coordinates of the chosen cluster
    - cluster_filenames: Image filename of the chosen cluster
    - cluster_central: The centroid of the chosen cluster
    '''
    def get_hist(fn):
        '''Get the color intersection between test image and train image fn.
            The higher the number of intersections, the better match.
        '''
        curr_img = cv2.imread('./train/' + fn + '.jpg')
        d_hist = cv2.calcHist([curr_img],[0],None,[256],[0,256])
        return cv2.compareHist(hist,d_hist,cv2.HISTCMP_INTERSECT)
    
    cluster_centrals = None
    cluster_elems = []
    cluster_filename = []
    cluster_hist = []
    cluster_count = []
    for i,coord in enumerate(data):
        # Adding the first point as the first cluster central
        if cluster_centrals is None:
            cluster_centrals = np.array([coord])
            cluster_elems.append([coord])
            cluster_filename.append([fnames[i]])
            compared_hist = get_hist(fnames[i])
            cluster_hist.append(compared_hist)
            cluster_count.append(1)
            continue
        # Get distance from point to each cluster
        distances = np.sum((cluster_centrals - coord)**2, axis=1)**0.5
        nearest = np.argmin(distances)
        # If point is far away from clusters, it's on its own cluster
        if distances[nearest] > max_range:
            # Stop when max number of clusters reached and have a big enough cluster
            if cluster_centrals.shape[0] >= max_clusters and max_clusters > 0:
                if np.max(cluster_count) >= min_size: break
                # Not big enough clusters means that the CNN is messed up
                return None,None,None
            cluster_centrals = np.append(cluster_centrals,[coord], axis=0)
            cluster_elems.append([coord])
            cluster_filename.append([fnames[i]])
            compared_hist = get_hist(fnames[i])
            cluster_hist.append(compared_hist)
            cluster_count.append(1)
        # If not, it belongs to cluster with nearest centeal. Update that one
        else:
            cluster_centrals[nearest] = (cluster_centrals[nearest] 
                                       * cluster_count[nearest] 
                                       + coord) / (cluster_count[nearest]+1)
            cluster_elems[nearest].append(coord)
            cluster_filename[nearest].append(fnames[i])
            compared_hist = get_hist(fnames[i])
            if compared_hist > cluster_hist[nearest]: cluster_hist[nearest] = compared_hist
            cluster_count[nearest] += 1
    biggest_cluster = np.argmax(cluster_count)
    similar_hist_cluster = np.argmax(cluster_hist)
    best_cluster = 0 if biggest_cluster == 0 and not take_best_hist else similar_hist_cluster
    # Return the coordinates, filenames, and center of the best cluster
    return cluster_elems[best_cluster], \
           cluster_filename[best_cluster], \
           cluster_centrals[best_cluster]

def displacement_calculation(test_img, centroid, coords, fnames, cam_matrix, 
                             ratio=0.6, max_range=3, max_displacement=10):
    '''Given a training cluster, predict the location of the test image

    Params
    ---
    - test_img: the test image
    - centroid: the current cluster centroid, for imputation in case of insufficient data
    - coords: coordinates of images in cluster
    - fnames: file names of images in cluster
    - cam_matrix: Camera intrinsic matrix K
    - ratio: Matching ration threshold to keep as per Lowe's Ratio test
    - max_range: Max distance from centroid to be considered within a cluster
    - max_displacement: Max distance from centroid to be considered valid result

    Returns
    ---
    loc: The final prediction of the coordinates
    '''
    if len(fnames) < 2: return centroid

    sift = cv2.SIFT_create()
        
    # FLANN parameters and initialize
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    kp_test,des_test = sift.detectAndCompute(test_img,None)
    
    kp_train = []
    des_train = []
    train_vecs = []
    cluster_centrals = None
    cluster_count = []

    for train in fnames:
        train_img = cv2.imread('./train/' + train + '.jpg')
        kp,des = sift.detectAndCompute(train_img,None)
        kp_train.append(kp)
        des_train.append(des)

        # Matching descriptor using KNN algorithm
        matches = flann.knnMatch(des,des_test,k=2)

        # Store all good matches as per Lowe's Ratio test.
        good = [m for m,n in matches if m.distance < ratio*n.distance]
        # Need to have 8 points to do the 8-point algorithm
        if len(good) < 8: 
            train_vecs.append(None)
            continue
        pts_train = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts_test = np.float32([kp_test[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        # Get translation matrix from essential matrix
        E,_ = cv2.findEssentialMat(pts_train,pts_test,cam_matrix,method=cv2.FM_LMEDS)
        _,_,T = cv2.decomposeEssentialMat(E)
        # Get x and z axis only, since y is assumed to be the same across pictures
        train_vecs.append(T[[0,2]])


    # For each pair, get all predicted positions and cluster them
    for pt1,pt2 in combinations(range(len(fnames)),2):
        # In case too few match, or duplicate translation vector, skip
        if train_vecs[pt1] is None or \
        train_vecs[pt2] is None or \
        np.allclose(train_vecs[pt1],train_vecs[pt2]): continue
        # Vector D
        displacement = (coords[pt2] - coords[pt1]).reshape(2,1)
        # Vector V: vertical stacking of 2 unit vectors v1,-v2
        unit_vectors = np.append(train_vecs[pt1],train_vecs[pt2], axis=1)
        # Final guard in case allclose doesn't work properly
        if np.linalg.det(unit_vectors) < 1e-4: continue
        # Solve this matrix and get b: V[b,c]' = D
        const = np.linalg.solve(unit_vectors,displacement)[0,0]
        # Vector b*v1 goes from Pt1 to Pt_test: Pt_test = Pt1 + b*v1
        loc = coords[pt1] + const * train_vecs[pt1].flatten()
        if cluster_centrals is None:
            cluster_centrals = np.array([loc])
            cluster_count.append(1)
            continue
        # Get distance from point to each cluster
        distances = np.sum((cluster_centrals - loc)**2, axis=1)**0.5
        nearest = np.argmin(distances)
        # If point is far away from clusters, it's on its own cluster
        if distances[nearest] > max_range:
            cluster_centrals = np.append(cluster_centrals,[loc], axis=0)
            cluster_count.append(1)
        # If not, it belongs to cluster with nearest centeal. Update that one
        else:
            cluster_centrals[nearest] = (cluster_centrals[nearest] 
                                       * cluster_count[nearest] 
                                       + loc) / (cluster_count[nearest]+1)
            cluster_count[nearest] += 1

    if cluster_centrals is None: return centroid
    # If things go well, take the closest cluster centroid to the initial pred
    cluster_distances = np.sum((cluster_centrals - centroid)**2, axis=1)**0.5
    nearest = np.argmin(cluster_distances)
    # If they are too far away or too inconsistent, SIFT may have been rigged, so cancel that
    if cluster_distances[nearest] > max_displacement or np.max(cluster_count)==1: return centroid
    return cluster_centrals[nearest]