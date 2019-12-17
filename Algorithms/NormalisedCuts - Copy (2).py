import numpy as np
import cv2
from scipy.sparse.linalg import eigsh


def calc_n_cut(eig_vec, cutVal, weights, twiddle_factor = 1e-12):
    ind_reg_a = np.where(eig_vec > cutVal)
    ind_reg_b = np.where(eig_vec <= cutVal)

    cut_a_b = np.sum(weights[ind_reg_a[0]][:, ind_reg_b[0]])
    assoc_a = np.sum(weights[ind_reg_a[0]])
    assoc_b = np.sum(weights[ind_reg_b[0]])
    return (cut_a_b / (assoc_a+twiddle_factor)) + (cut_a_b / (assoc_b+twiddle_factor))


def perform_a_normalised_cut(image_indc, _weights, thresh, search_point_count=20, k=2):
    d = np.sum(_weights, axis=0)
    D = d*np.eye(_weights.shape[0])
    D_inv = (1/d)*np.eye(_weights.shape[0])

    eig_val, eig_vec = eigsh((D-_weights), k=k, Minv=D_inv, M=D, which='SM', tol=0.001)

    for index in range(1, len(eig_vec.T)):
        search_points = np.linspace(np.min(eig_vec.T[index]), np.max(eig_vec.T[index]), search_point_count+1)[:-1]

        n_cut_values = list(map(lambda cut_val: calc_n_cut(eig_vec.T[1], cut_val, _weights), search_points))

        best_n_cut_ind: int = np.argmin(n_cut_values)
        worst_n_cut_ind: int = np.argmax(n_cut_values)
        if n_cut_values[worst_n_cut_ind] - n_cut_values[best_n_cut_ind] >= 0.06:
            break
    else:
        print(n_cut_values[best_n_cut_ind])
        return None, None, None, None
    if n_cut_values[best_n_cut_ind] > thresh:
        print(n_cut_values[best_n_cut_ind])
        return None, None, None, None

    print(n_cut_values[best_n_cut_ind])

    cut_val = search_points[best_n_cut_ind]
    ind_reg_a = np.where(eig_vec.T[index] > cut_val)[0]
    ind_reg_b = np.where(eig_vec.T[index] <= cut_val)[0]

    return image_indc[ind_reg_a], image_indc[ind_reg_b],\
           _weights[ind_reg_a][:, ind_reg_a], _weights[ind_reg_b][:, ind_reg_b]


def calculate_weights(img_flat, img_ind, r_sq=5**2, sigma_x_sq=3**2, sigma_i_sq=7**2):

    square_dist = np.sum(np.square((img_ind[:, None] - img_ind)), axis=2)
    ind_hold = np.where(square_dist < r_sq)
    weights = np.zeros(square_dist.shape)
    weights[ind_hold] = -square_dist[ind_hold] / sigma_x_sq

    square_dist = np.sum(np.square(img_flat[ind_hold[0]] - img_flat[ind_hold[1]]), axis=1)
    weights[ind_hold] -= square_dist / sigma_i_sq
    weights[ind_hold] = np.exp(weights[ind_hold])
    return weights


def recursive_normalised_cuts(image_indc, _weights, thresh, search_point_count=20, region_min=10, k=2):
    if len(image_indc) < region_min:
        return [image_indc]
    ind_a, ind_b, weights_a, weights_b = perform_a_normalised_cut(image_indc, _weights, thresh, search_point_count, k)
    if ind_a is None:
        return [image_indc]

    return recursive_normalised_cuts(ind_a, weights_a, thresh, search_point_count, k) + \
        recursive_normalised_cuts(ind_b, weights_b, thresh, search_point_count, k)


def normalised_cuts(img, thresh=0.04, r_sq=10**2, sigma_x_sq=5**2, sigma_i_sq=0.1**2, search_point_count=20,
                    region_min=10, k=2):
    scale = 5000.0/(img.shape[0]*img.shape[1])
    if scale < 1.0:
        srt_scale = np.sqrt(scale)
        scaled_img = cv2.resize(img, (0, 0), fx=srt_scale, fy=srt_scale)
    else:
        scaled_img = img

    #img_min = np.min(scaled_img)
    #scaled_img = (scaled_img - img_min) / (np.max(scaled_img) - img_min)

    img_flat_features = cv2.cvtColor((img/255).astype(np.float32), cv2.COLOR_BGR2HSV_FULL).reshape((-1, scaled_img.shape[-1]))
    s = img_flat_features[:, 1].copy()
    img_flat_features[:, 1] = img_flat_features[:, 2]*s*np.sin(img_flat_features[:, 0] * np.pi / 180)
    img_flat_features[:, 0] = img_flat_features[:, 2]*s*np.cos(img_flat_features[:, 0] * np.pi / 180)
    ind = np.indices(scaled_img.shape[:2]).reshape((2, -1)).T
    weights = calculate_weights(img_flat_features, ind, r_sq, sigma_x_sq, sigma_i_sq)

    image_ind = np.arange(len(weights))
    class_list = recursive_normalised_cuts(image_ind, weights, thresh, search_point_count, region_min, k)

    img_out = np.zeros(scaled_img.shape[:-1])
    for class_id, color_class_ind in enumerate(class_list):
        img_out.flat[color_class_ind] = class_id
    if scale < 1.0:
        img_out = cv2.resize(img_out, img.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST)
    return img_out, len(class_list)

# img = cv2.imread('../Datasets/house.PNG').astype(np.uint8)
# img = cv2.imread('../Datasets/aaaaaa - Copy.bmp').astype(np.int64)
# classes_img, num_classes = normalised_cuts(img, r_sq=10000**2, sigma_x_sq=200**2)
# classes_img, num_classes = normalised_cuts(img, thresh=0.3, r_sq=5**2, sigma_x_sq=2**2, sigma_i_sq=0.1**2)


img = cv2.imread('../Datasets/BSDS300/images/train/176035.jpg')
classes_img, num_classes = normalised_cuts(img)

disp_img = np.zeros(img.shape)
for class_id in range(num_classes):
    ind_c = np.where(classes_img == class_id)
    class_avg_int = np.average(img[ind_c], axis=0)
    disp_img[ind_c] = class_avg_int
cv2.imshow("img", img.astype(np.uint8))
cv2.imshow("disp_img", disp_img.astype(np.uint8))
cv2.waitKey(0)




