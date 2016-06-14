#!/usr/bin python
# Sat Jun 11 12:36:57 2016

from __future__ import division

## Imports
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import Tkinter as tk
from PIL import ImageTk, Image # help for use of .tif with Tkinter
from scipy import stats # to compute Chi-Square RV
from scipy import linalg # for rigid transform
import cPickle as pickle

try:
    g_means = pickle.load(open('g_means.p', 'rb'))
    g_inv_covs = pickle.load(open('g_inv_covs.p', 'rb'))
except:
    pass

####### Loading Files
def load_file(filename):
    """Use to load landmarks"""
    array = []
    with open(filename) as f:
        for line in f:
            array.append(float(line))
    return np.array(array)

def load_image(filename):
    img = cv2.imread(filename)
    img_g = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    img_g[:,:] = img[:,:,0]
    return img, img_g

def plot_shape(shapes, n=40, **kwargs):
    for shape in shapes:
        shape = shape.reshape((n, 2))
        plt.scatter(shape[:,0], shape[:,1], **kwargs)
    plt.axis('equal')

def load_landmarks(type='1'):
    
    def reform_filename():
        def wrapper(filename):
            if filename[-8] == '1':
                som = 10
            else:
                som = 0
            return som + int(filename[-7])
        return wrapper
        
    # get all incisor from top left
    landmark_files = []
    for filename in os.listdir('Data/Landmarks/original'):
        if filename[-3:] == 'txt' and filename[-5] == type: # 1 = top-left incisor
            landmark_files.append('Data/Landmarks/original/'+filename)
    landmark_files.sort(key=reform_filename())
    
    landmarks = []
    for filename in landmark_files:
        array = []
        with open(filename) as f:
            for line in f:
                array.append(float(line))
        landmarks.append(array)
    return np.array(landmarks)

####### Building an Active Shape Model

def center(image):
    c = np.mean(image, axis=0)
    return image.copy() - c , c

def sum_of_squares(vector):
    return np.sqrt(sum(vector**2))

def align_shape(shape1, shape2, weights=None):
    """Align shape2 to shape2, using affine transform"""
    new_shape = shape2.copy()
    theta, s, t = procrustes(shape1.copy(), new_shape, weights)
    
    # Construct rotation and translation matrix (M and t)
    M = np.array([[s*np.cos(theta), -s*np.sin(theta)],
                  [s*np.sin(theta),  s*np.cos(theta)]]).reshape(2,2)
    t = np.reshape(t, (1,2))
    
    new_shape = np.dot(new_shape - t, M) + t
    return new_shape

def procrustes(shape1, shape2, weights=None):
    """Returns rotation, scale and translation to best align shape2 with shape1
    input: shape1, shape2, weights = weight matrix
    returns: theta, scale, translation"""
    
    # 1. Translation normalization => centered configuration    
    init1 = np.mean(shape1, axis=0)
    init2 = np.mean(shape2, axis=0)
    A = shape2 - init2
    B = shape1 - init1
    
    # 2. Scale normalization => Pre-Shape
    SSQ_A = linalg.norm(A)
    SSQ_B = linalg.norm(B)
    A = A / SSQ_A
    B = B / SSQ_B
    
    # 3. Rotation normalization => Shape
    T1 = np.dot(A.T.dot(B), (A.T.dot(B)).T)
    T2 = np.dot((A.T.dot(B)).T, A.T.dot(B))
    V, _, _ = linalg.svd(T1)
    W, _, _ = linalg.svd(T2)
    #print 'norm(shape1) = %f, norm(shape2) = %f' % (SSQ_A, SSQ_B)
    R = np.dot(V, W.T)
    theta = np.arctan(R[1,0] / R[0,0])
    return theta, SSQ_B/SSQ_A, init2

def generalized_procrustes(shapes, epsilon=1e-2):
    """Align all shapes"""
    try:
        m, n = shapes[0,:].shape
    except:
        m = shapes[0,:].shape[0]
        n = 1
    new_shapes = []
    for shape in shapes:
        new_shape = np.reshape(shape, (m*n/2, 2))
        new_shape = new_shape - np.mean(new_shape, axis=0) # center them
        new_shapes.append(new_shape)

    mean_shape = np.sum(new_shapes, axis=0) / len(new_shapes)
    # (re)allign the mean shape with the first shape
    mean_shape = align_shape(new_shapes[0], mean_shape)
    converged = False
    while not converged:
        new_shapes = [align_shape(mean_shape, shape) for shape in new_shapes]
        new_mean = np.sum(new_shapes, axis=0) / len(new_shapes)
        # (re)allign the mean shape with the first shape
        new_mean = align_shape(new_shapes[0], new_mean)
        if np.mean((new_mean-mean_shape)**2) < epsilon:
            converged = True
        mean_shape = new_mean
    return np.array([shape.flatten() for shape in new_shapes])

####### Principal Components Analysis

def PCA(landmarks, p=1.0, keepall=False):
    """ Perform Principal Component Analysis
    input:   landmarks: set of landmarks on which PCA is to be performed
             p: fraction of variance to be explained
             keepall: if True, keep all eigenvectors, irrespective of p
    returns: bs: transformed landmarksin new coordinate system
             P: matrix of eigenvectors
             mean_image: mean_image
             V: vector of eigenvalues
    """
    # 1. Get mean image
    mean_image = np.mean(landmarks, axis=0)[None,:]
    landmarks_normal = landmarks - mean_image

    # 2. Compute covariance matrix
    cov_matrix = np.cov(landmarks_normal.T)

    # 3. Compute eigenvalues / vectors
    V, D = np.linalg.eigh(cov_matrix)
    idx = np.argsort(np.abs(V))[::-1] # Get indexes by magnitude of eigenvalue

    # 4. Keep P% of energy
    Vcum = np.cumsum(V[idx])
    Vcum /= Vcum[-1]
    if keepall:
        N = len(V)
    else:
        N = np.where(Vcum < p)[-1][-1] # number of eigenvalues to keep
    D_reduced = D[:,idx[:N]]
    P = D_reduced[:] # To stay consistent with notation
    V_reduced = V[idx[:N]]

    # 5. Calculate b's (Eq. 3 of asm_overview.pdf)
    bs = np.dot(P.T, landmarks_normal.T)
    
    return bs, P, mean_image, V_reduced

####### Histogram Equalization
def histeq(img, nbr_bins=256):
    """Histogram equalization of a grayscale image"""
    # get histogram
    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf_normalized = cdf * imhist.max() / cdf.max() # normalize
    
    # use linear interpolation of cdf to find new pixel values
    img2 = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    
    return img2.reshape(img.shape), cdf

####### Fit the model to an image

def scale(image, s=1.0):
    """Scale an image (landmarks) with factor s."""
    new_image = image.copy()
    return new_image * s

def rotate(image, theta=0.0):
    """Rotate an image (landmarks) with angle theta."""
    new_image = image.copy()
    try:
        m, n = np.shape(new_image)
    except ValueError:
        m = np.shape(new_image)
        new_image = np.reshape(new_image, (m[0]/2, 2))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    new_image = np.dot(R, new_image.T)
    return new_image.T

def normalize(direction):
    """Set length of direction vector to 1"""
    length = np.sqrt(sum([x**2 for x in direction]))
    if length == 0:
        return x
    return [x/float(length) for x in direction]

def translate(image, x=0., y=0.):
    new_image = image.copy()
    return (new_image.T + np.array([x,y]).reshape(2,1)).T    

def get_landmarks_profile(image, landmarks, N=5, show=False):
    """ Implement a procedure to move each model point separately toward the new (desired?) position
     As described in asm 4.1  
     For each point: look around points, in direction normal to boundary
     N: determines the neighbourhood where is searched
     show: if True, shows image and overlying points and directions"""
    
    if show:
        plt.figure(figsize=(10,10))
        plt.imshow(image, cmap='gray'); plt.hold(True)    
    
    m, n = image.shape
    profile = {}    # store values of 'image' in specific direction
    coords  = {}    # store the coordinates of these directions
    for i in range(len(landmarks[:,0])):
        if show: plt.scatter(landmarks[i,0], landmarks[i,1], marker='o', color='r')
        # Determine direction perpendicular to edges of landmarks
        direction = ((landmarks[i,1] - landmarks[i-1,1] ), -(landmarks[i,0] - landmarks[i-1,0]))
        direction = normalize(direction)
        profile[i] = []
        coords[i] = []
        for k in range(-N, N+1):
            x = int(landmarks[i,0] + k*direction[0])
            y = int(landmarks[i,1] + k*direction[1])
            #print k, x, y, m, n, new_image[i,0], new_image[i,1] 
            if 0 < x < n and 0 < y < m:
                if show: plt.scatter(x, y, marker='.', s=1)
                coords[i].append((x, y))
                profile[i].append(image[y, x])
    return coords, profile


def average(vector):
    average = []
    vector = [float(v) for v in vector]
    for i in range(1,len(vector)-1):
        average.append(.5*(vector[i+1] + vector[i-1]))
    return average

def gradient(vector):
    gradient = []
    vector = [float(v) for v in vector]
    for i in range(len(vector)-1):
        gradient.append(vector[i+1] - vector[i])
    return gradient

def find_next(profile, window_size=4):
    """For a profile, find the best edge """
    x = np.array(profile)
    try:
        y = np.convolve(x, np.ones((window_size,)), mode='valid')
    except ValueError:
        return 0, 0
    magnitude = x[np.argmax(np.gradient(y))] # edge strength e_max
    return np.argmax(np.gradient(y)), np.abs(magnitude)

def find_next2(profile, landmark=0):
    """For a profile, find optimal displacement
    according to the procedure in asm_overview.pdf 4.3"""
    x = np.array(profile)
    g = np.gradient(x)
    g = g / np.sum(g)
    
    incissor = G.incissor
    g_mean = g_means[incissor, landmark]
    Sinv = g_inv_covs[incissor, landmark]
    fit_costs= []  
    for pos in range(len(g) - len(g_mean)):
        g_temp = g[pos:pos+len(g_mean)]
        fit_costs.append(np.dot(np.dot((g_temp-g_mean).T, Sinv), g_temp-g_mean))
    return int(np.argmin(fit_costs) + len(g_mean)/2) + 1
    
def adjust_shape(image, landmarks, centrum=None, n_iters=10, weights=None, method=1):
    """This function adjusts the shape of the landmarks, both in pose and shape parameters.
    inputs:
        image: the grayscale image on wich the shape has to fitted
        landmarks: the set of (original) landmarks that will be reshaped
        centrum: the original position of the center of the landmarks
        n_iters: number of iterations
        weights: set of weights, wich determine imprtance during param update
                    if None -> default to all weights == 1.0
        method: select method to find X+dX form:
                    if 1 -> gradient method from original paper by Cootes
                    if 2 -> method based on statistics of training images, from asm_overview.pdf
    """
    
    if weights == None:
        weights = {'t' : 1.0,
                   'theta' : 1.0,
                   's' : 1.0,
                   'b' : np.ones((80, 1))}
    
    landmarks = generalized_procrustes(landmarks)
    bs, P, mean_shape, V = PCA(landmarks, keepall=True)
    
    
    # if necessary: give correct form to eigenvalues + threshold if too small
    V = np.reshape(V, (len(V),1))
    V[ np.abs(V) < 1e-10] = 0
    m, n = mean_shape.shape
    mean_shape = mean_shape.reshape((m*n/2, 2))# - np.array([left, top]) # adjust for cropped image
    
    if centrum == None:
        Xc, Yc = (G.trans_x, G.trans_y)
    else:
        Xc, Yc = centrum                # position of center of model
    theta, s = 0.0, 1.0             # initial values of theta and s
    
    # initial values -> start with mean shape
    x = mean_shape.copy() 
    X = mean_shape.copy() + np.array([Xc, Yc]).reshape(1,2)

    for jj in range(n_iters):
        coords, profile = get_landmarks_profile(image, X.copy(), N=20, show=False)
        
        if method == 1:
            # Procedure 1 - According to original paper
            dX = np.zeros((40, 2))
            mags = np.zeros((40, 1))
            for i in profile.keys(): # point with which we're working
                next_point, magnitude = find_next(profile[i], window_size=3)
                dX[i, 0] = coords[i][ next_point ][0] - X[i, 0]
                dX[i, 1] = coords[i][ next_point ][1] - X[i, 1]
                mags[i] = magnitude
            
            mags = mags / np.max(mags)
            X_dX = X + dX * mags
        
        elif method == 2:
            # Procedure 2 - According to asm_overview.pdf
            dX = np.zeros((40, 2))
            for i in profile.keys():
                next_point = find_next2(profile[i], landmark=i)
                dX[i, 0] = coords[i][ next_point ][0] - X[i, 0]
                dX[i, 1] = coords[i][ next_point ][1] - X[i, 1]
            X_dX = X + dX

        # Calculate initial s, theta, t for current image: X = Mx + Xc

        # Construct rotation and translation matrix (M and t)
        #   1. compute y
        M = np.array([[s*np.cos(theta), -s*np.sin(theta)],
                      [s*np.sin(theta),  s*np.cos(theta)]]).reshape(2,2)
                       
        M_inv = np.array([[1.0/s*np.cos(-theta), -1.0/s*np.sin(-theta)],
                       [1.0/s*np.sin(-theta),  1.0/s*np.cos(-theta)]]).reshape(2,2)
        
        dXc, dYc = np.mean(X_dX, axis=0) - np.mean(X, axis=0)
        x_dx = np.dot(X_dX - np.array([Xc+dXc, Yc+dYc]), M_inv)
        dtheta, ds, dt = procrustes(x_dx, x)

        dx = x_dx - x
        db = np.dot(P.T, dx.reshape((80, 1)))
        b  = np.dot(P.T, (x - mean_shape).reshape((80, 1)))
        
        ## update parameters
        Xc = Xc + weights['t']*dXc                  # Eq 22
        Yc = Yc + weights['t']*dYc                  # Eq 23
        theta = theta + weights['theta']*dtheta     # Eq 24
        s = s * weights['s']*ds                     # Eq 25
        b = b + np.multiply(weights['b'], db)       # Eq 26

        ### Limit values of b
        ## Approach 1: limit db[i] according to eq. 15
        # if b + db < 3*sqrt(eigenval) -> set to max val
        max_dev = lambda x: 3*np.sqrt(x)
        for i in range(len(b)):
            if np.abs(b[i]) > max_dev(V[i]):
                b[i] =  np.sign(b[i]) * max_dev(V[i])
        
        ## Approach 2: is deformed shape acceptable? eq. 16
        
#        # Dmax is chosen according to chi-squared distribution
#        # where # degrees of freedom = size of db
#        rv = stats.chi2(df=np.shape(P)[1])
#        accept_ratio = .99 # ratio of shapes to accept
#        Dmax = rv.ppf(accept_ratio)
#        Dm   = 0.0
#        for i in range(np.shape(P)[1]):
#            Dm += np.abs(b[i])**2 / V[i]
#        if Dm > Dmax:
#            print 'shape unacceptable'
#            b = np.zeros(b.shape)

        # reconstruct, based on updated values
        
        x = np.dot(P, b) 
        x = np.reshape(x, (40, 2)) + mean_shape

        # new M
        M = np.array([[s*np.cos(theta), -s*np.sin(theta)],
                      [s*np.sin(theta),  s*np.cos(theta)]]).reshape(2,2)        
        
        Xnew = np.dot(x, M) + np.array([Xc, Yc]).reshape(1,2)
        if np.linalg.norm(X - Xnew) < 1e-2:
            print 'Converged'
            return Xnew
        X = Xnew

    return X

def show_image(image, X):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap='gray'); plt.hold(True)
    plot_shape([X],  c='y') 

def find_edges(image):
    th, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(image, th/2, th)
    return edges


## Tkinter methods

class G:
    """Class for global variables."""
    trans_x = 617# 588
    trans_y = 341# 307
    root = None
    incissor = 1
    best_init = {1: (368, 394), # best initial values for centre of first shape
                 2: (360, 350),
                 3: (408, 361),
                 4: (371, 385),
                 5: (410, 380),
                 6: (399, 313),
                 7: (373, 362),
                 8: (419, 306),
                 9: (420, 447),
                 10: (357, 279),
                 11: (308, 346),
                 12: (399, 394),
                 13: (374, 239),
                 14: (332, 409)}
    
def motion(event):
    x, y = event.x, event.y
    print('{}, {}'.format(x, y))

def mouse_down(event):
    print "frame coordinates: %s/%s" % (event.x, event.y)
    print "root coordinates: %s/%s" % (event.x_root, event.y_root)
    G.trans_x = event.x
    G.trans_y = event.y
    G.root.destroy()

def find_all(image, start_point=None, weights=None, method=1):
    """Find all inscissors in the image. This is done by looking for all
    the teeth centers, based on the first (top left) tooth, which must be
    set with the mouse.
    """    
    
    results = {}
    center_points = {}
    
    if start_point == None:
        cx, cy = G.trans_x, G.trans_y
    else:
        cx, cy = start_point

    profile = image[cy, cx:cx+400] # 300? good value?
    y = np.convolve(profile, np.ones((5,)), mode='same')

    plt.figure(figsize=(10,10));
    img_sub = image[400:500, cx:cx+300]
    plt.figure(figsize=(10,10));plt.imshow(img_sub, cmap='gray');plt.plot([0, 300], [30, 30])
    z = np.convolve(np.gradient(y), np.ones((5,)), mode='valid')
    
    # determine horizontal transistion points 
    z = np.gradient(z)
    plt.plot(z)
    sep_1 = cx + np.argmax(z[:100])
    sep_2 = cx + 100 + np.argmax(z[101:200])
    sep_3 = cx + 200 + np.argmax(z[201:300])
    print 'Separators: ', sep_1, sep_2, sep_3, sep_3 + 2.0*(sep_1 - cy)
    plt.scatter([sep_1 - cx, sep_2 - cx, sep_3 - cx],
                 [30, 30, 30], c='y')
    
    center_points[1] = (cx, cy)
    center_points[2] = (sep_1 + 0.5*(sep_2 - sep_1), cy)
    center_points[3] = (sep_2 + 0.5*(sep_3 - sep_2), cy)
    center_points[4] = (sep_3 + 0.5*(sep_3 - sep_2), cy)

    new_line = image[cy:cy+300, sep_2]
    plt.figure(figsize=(10,10))
    plt.plot(np.convolve(new_line, np.ones((8,)), mode='valid'))
    plt.title('min @ %d' % np.argmin(np.convolve(new_line, np.ones((8,)), mode='valid')))
    
    cy2 = cy + 2.0*np.argmin(np.convolve(new_line, np.ones((8,)), mode='valid'))
    profile = image[cy2, sep_2-150:sep_2+150]
    y = np.convolve(profile, np.ones((5,)), mode='same')
    z = np.convolve(np.gradient(y), np.ones((5,)), mode='valid')
    plt.figure(figsize=(10,10))
    plt.plot(z)
    plt.title('Horizontal second derivative')
    
    sep_4 = sep_2 - 150 + np.argmin(z[:100])
    sep_5 = sep_2 - 150 + 100 + np.argmin(z[101:200])
    sep_6 = sep_2 - 150 + 200 + np.argmin(z[201:-10])
    width = sep_5 - sep_4
    print 'sep_4, sep_5, sep_6 = ', sep_4, sep_5, sep_6
    
    center_points[5] = (sep_4 - 0.5*width, cy2)
    center_points[6] = (sep_5 - 0.5*width, cy2)
    center_points[7] = (sep_5 + 0.5*width, cy2)
    center_points[8] = (sep_6 + 0.5*width, cy2)
    
    for inscissor_type in range(1,9):
        
        G.incissor = inscissor_type
        landmarks = load_landmarks(type=str(inscissor_type))
        results[inscissor_type] = adjust_shape(image, landmarks, 
                                        centrum=center_points[inscissor_type], 
                                        n_iters=64, weights=weights,
                                        method=method)

    # plot all landmarks
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='gray'); plt.hold(True)
    plot_shape([results[i] for i in range(1,9)], s=1, color='y')
    for c in center_points.values():
        plt.scatter(c[0], c[1])
    plt.scatter(sep_2, cy2, c='r')
    plt.title('Adjusted')
    plt.axis('tight')
    plt.show()
    
    return results

def align(shape1, shape2):
    """Align in least-square fashion, shape2 to shape1
    See: http://www.cse.psu.edu/~rtc12/CSE586/lectures/cse586Shape1.pdf"""
    # 1. Translation normalization => centered configuration    
    init1 = np.mean(shape1, axis=0)
    init2 = np.mean(shape2, axis=0)
    A = shape2 - init2
    B = shape1 - init1
    
    # 2. Scale normalization => Pre-Shape
    SSQ_A = linalg.norm(A)
    SSQ_B = linalg.norm(B)
    A = A / SSQ_A
    B = B / SSQ_B
    
    # 3. Rotation normalization => Shape
    T1 = np.dot(A.T.dot(B), (A.T.dot(B)).T)
    T2 = np.dot((A.T.dot(B)).T, A.T.dot(B))
    V, _, _ = linalg.svd(T1)
    W, _, _ = linalg.svd(T2)
    #print 'norm(shape1) = %f, norm(shape2) = %f' % (SSQ_A, SSQ_B)
    R = np.dot(V, W.T)
    #print 'R = ', R
    return np.dot(A, R) * SSQ_A + init2

def create_PDMs(type='1'):
    
    ## Load all shapes
    landmarks = load_landmarks(type)
    
    ## Build an Active Shape Model
    
    # Step 1. center of gravity -> origin
    for k in range(14):
        landmarks[k,:] = center(landmarks[k,:].reshape(40, 2))[0].reshape(80)
    
    # Step 2. scale such that |x| = 1
    mean = landmarks[0,:].copy()
    mean /= sum_of_squares(mean)
    
    # Step (3 +) 4: Allign all shapes with the current estimate of the mean
    landmarks = generalized_procrustes(landmarks)
    
    ## Principal Components Analysis
    bs, P, mean_image, V = PCA(landmarks, p=.99, keepall=True)

#    plt.figure(figsize=(10,10))
#    plot_shape([center(landmarks)[0]], c='r')
#    plot_shape([center(temp)[0]])
#    plot_shape([center(new_landmarks)[0]], s=1)
#    plt.legend(['original', 'affine', 'goal'])
    
    m, n = P.shape
    for eig_vec in range(n):
        plt.figure(figsize=(10,10))
        plt.title('Variations for eigenvector %d' % eig_vec)
        dbs = np.linspace(-3*np.sqrt(V[eig_vec]), 3*np.sqrt(V[eig_vec]), 9)
        for db in dbs:
            b = np.zeros(n)
            b[eig_vec] = db
            new_image = mean_image + np.dot(P, b)
            if db == 0:
                plot_shape([new_image], label='db = %f' % db)
            else:
                plot_shape([new_image], s=1, label='db = %f' % db)
            plt.legend(loc=3)

 
def create_stat_model(input_file = 'gradients.p', save_files =  ('g_means.p', 'g_inv_covs.p')):
    """Create mean and inverse covariance matrix, for statistical model.
    input:
        input_file: file where gradients are pickled
        save_files: filenames where to store means and inv covs, resp.
    """
    gradients  = pickle.load( open( input_file, "rb" ) )
    g_means = {}
    g_inv_covs  = {}
    for incissor_type in [1,2,3,4,5,6,7,8]:
        for landmark in range(40):
            gs = np.zeros((14, 11))
            for image in range(1,15):
                gs[image-1,:] = gradients[incissor_type, image, landmark]
            g_means[incissor_type, landmark] = gs.mean(axis=0)
            try:
                g_inv_covs[incissor_type, landmark]  = np.linalg.inv(np.dot(gs.T, gs))
            except: # solve with SVD
                print 'Singular matrix...'
                c = np.dot(gs.T, gs)
                L, Q = np.linalg.eig(c)
                Q = Q[:,L>1e-10]
                L = L[L>1e-10]
                g_inv_covs[incissor_type, landmark] = np.dot(np.dot(Q, np.diag(L)), Q.T)
    pickle.dump(g_means, open(save_files[0], 'wb'))
    pickle.dump(g_inv_covs, open(save_files[1], 'wb'))
                
def create_gradients(save_file='gradients.p'):
    """Create all gradient vectors, for all landmarks and all images
    Can be used to create statistical model."""
    gradients = {}
    ## Learn the distribution of the training points around ...
    for incissor_type in [1,2,3,4,5,6,7,8] :
        landmarks = load_landmarks(type=str(incissor_type))
        for filename in os.listdir('./Data/Radiographs'):
            if filename[-3:] == 'tif':
                ## Prepare image
                idx = int(filename[-5]) + 10*int(filename[-6])
                # get part of image    
                left = 1000; top = 500; right = 2000; lower = 1500;
                image = Image.open('./Data/Radiographs/'+filename).convert('L')
                image = image.crop((left, top, right, lower))
                image = np.array(image)
                
                coords, profiles = get_landmarks_profile(image,
                                    landmarks[idx-1, :].reshape((40,2)) - np.array([left, top]),
                                    N=5, show=False)
                
                for key in profiles:
                    x = np.array(profiles[key])
                    y = np.gradient(x)
                    gradients[incissor_type, idx, key]  = y /  np.sum(y)
#                   print 'type=%d, landmark=%d, point=%d' % (incissor_type,
#                                                              idx,
#                                                              key)
    pickle.dump(gradients, open(save_file, 'wb'))
    return gradients

def MSE(X, Y):
    """Returns the mean-squared error between X and Y."""
    return np.mean((X - Y)**2)
    
