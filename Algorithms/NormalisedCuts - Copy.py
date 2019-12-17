import numpy as np
import cv2
from scipy.sparse.linalg import eigsh
from scipy.linalg import sqrtm

def calc_n_cut(eig_vec, cutVal, weights):
    ind_reg_a = np.where(eig_vec > cutVal)
    ind_reg_b = np.where(eig_vec <= cutVal)

    cut_a_b = np.sum(weights[ind_reg_a[0]][:, ind_reg_b[0]])
    assoc_a = np.sum(weights[ind_reg_a[0]])
    assoc_b = np.sum(weights[ind_reg_b[0]])
    return (cut_a_b / assoc_a) + (cut_a_b / assoc_b)



# img = cv2.imread('../Datasets/BSDS300/images/train/374067.jpg', cv2.IMREAD_GRAYSCALE).astype(np.int64)
img = cv2.imread('../Datasets/asdf.PNG').astype(np.int64)
img_min = np.min(img)
img = (img-img_min)/(np.max(img)-img_min)

# img2 = img+np.random.normal(30, 5, img.shape)
# img2 = img2.clip(0, 255)
# cv2.imwrite("../Datasets/test_noisy.bmp", img2)

img_flat = img.reshape((-1, img.shape[-1]))
ind = np.indices(img.shape[:2]).reshape((2, -1)).T

r_sq = 5**2
sigma_x_sq = 3**2
sigma_i_sq = 0.07**2

square_dist = np.sum(np.square((ind[:, None] - ind)/np.max(ind)), axis=2)
ind_hold = np.where(square_dist < r_sq)
weights = np.zeros(square_dist.shape)
weights[ind_hold] = np.exp(-square_dist[ind_hold]/sigma_x_sq)

square_dist = np.sum(np.square(img_flat[ind_hold[0]]-img_flat[ind_hold[1]]), axis=1)
weights[ind_hold] *= np.exp(-square_dist/sigma_i_sq)

d = np.sum(weights, axis=0)
D = d*np.eye(weights.shape[0])
D_inv = (1/d)*np.eye(weights.shape[0])
l=20


eig_val, eig_vec = eigsh((D-weights), Minv=D_inv, M=D, which='SM', tol=0.0001)

index = 1
for index in range(1, len(eig_vec.T)):
    search_points = np.linspace(np.min(eig_vec.T[index]), np.max(eig_vec.T[index]), l+1)[:-1]

    n_cut_vals = list(map(lambda cutVal: calc_n_cut(eig_vec.T[1], cutVal, weights), search_points))

    best_n_cut_ind = np.argmin(n_cut_vals)
    worst_n_cut_ind = np.argmax(n_cut_vals)
    if n_cut_vals[worst_n_cut_ind] - n_cut_vals[best_n_cut_ind] >= 0.06:
        break
else:
    print("warning split may be inaccurate as a good split index was not found")





cv2.imshow("img", (img*255).astype(np.uint8))
cutVal = search_points[best_n_cut_ind]
stepVal=0.001

while True:

    ind_reg_a = np.where(eig_vec.T[index] > cutVal)

    img_out = np.ones(img.shape)*[0,0,1]
    img_out.reshape((-1, img.shape[-1]))[ind_reg_a] = img_flat[ind_reg_a]

    cv2.imshow("img_out", (img_out*255).astype(np.uint8))
    key = cv2.waitKey(0)
    if key == 27:
        break
    if key == ord('+'):
        cutVal += stepVal
        print("Cut val now: "+str(cutVal))
    if key == ord('-'):
        cutVal -= stepVal
        print("Cut val now: "+str(cutVal))
    if key == ord('0'):
        stepVal *= 10
        print("Step val now: "+str(stepVal))
    if key == ord('9'):
        stepVal /= 10
        print("Step val now: "+str(stepVal))
    if key == ord('1'):
        index = index-1 if index > 0 else index
        print("index val now: "+str(index))
    if key == ord('2'):
        index = index+1 if index < len(eig_vec.T)-1 else index
        print("index val now: "+str(index))
