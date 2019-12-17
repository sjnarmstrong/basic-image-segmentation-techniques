import numpy as np
from scipy.stats import multivariate_normal
from math import ceil
import cv2


class Distribution:
    def __init__(self, data, k):
        #datalen = len(data)
        #random_subset_data = data[np.random.randint(0, datalen, ceil(datalen/k))]
        #cov = np.cov(data.T)
        data_dim = len(data[0])
        mean = np.random.uniform(np.min(data, axis=0), np.max(data, axis=0))
        var = np.max(data, axis=0) - np.min(data, axis=0)
        self.dist = multivariate_normal(mean, np.eye(data_dim)*var)
        self.prior = 1.0/k

    def calc_expectation(self, data):
        return self.dist.pdf(data)*self.prior

    def maximise_dist(self, data, weights, expectations):
        self.dist.mean = np.sum((data.T*weights).T, axis=0)
        dist_from_mean = data-self.dist.mean
        self.dist.cov = np.dot(dist_from_mean.T*weights, dist_from_mean)
        self.prior = np.average(expectations)


def perform_em(X, k, thresh=1e-12, m=1000000):
    distributions = [Distribution(X, k) for i in range(k)]

    expectations = np.zeros((len(distributions), len(X)))
    for i, d in enumerate(distributions):
        expectations[i] = d.calc_expectation(X)
    scale_exp = np.sum(expectations, axis=0)
    log_likelihood = np.sum(np.log(scale_exp))

    for j in range(m):

        expectations /= scale_exp
        weights = (expectations.T/np.sum(expectations, axis=1)).T

        for i, d in enumerate(distributions):
            d.maximise_dist(X, weights[i], expectations[i])
            expectations[i] = d.calc_expectation(X)

        scale_exp = np.sum(expectations, axis=0)
        log_likelihood_curr = np.sum(np.log(scale_exp))
        print(log_likelihood/log_likelihood_curr)

        if log_likelihood/log_likelihood_curr < 1.0+thresh:
            break
        log_likelihood = log_likelihood_curr

    return expectations / scale_exp


def em_image_seperation(img, k, thresh=1e-7, max_it=1000000):
    img_data = cv2.cvtColor(img, cv2.COLOR_BGR2Luv).reshape((-1, img.shape[-1]))
    expectations = perform_em(img_data, k)

    img_out = np.zeros(img.shape[:-1])
    img_out.flat[:] = np.argmax(expectations, axis=0)
    return img_out, k


def em_image_seperation_with_spatial_info(img, k, thresh=1e-7, max_it=1000000):
    img_data = cv2.cvtColor(img, cv2.COLOR_BGR2Luv).reshape((-1, img.shape[-1]))
    ind = np.indices(img.shape[:2]).reshape((2, -1)).T
    img_data = np.hstack((img_data, ind))
    expectations = perform_em(img_data, k)

    img_out = np.zeros(img.shape[:-1])
    img_out.flat[:] = np.argmax(expectations, axis=0)
    return img_out, k


def get_color_image_from_classes(k, classes_img, img):
    out_img = np.zeros(img.shape)
    for class_id in range(k):
        ind_c = np.where(classes_img == class_id)
        class_avg_int = np.average(img[ind_c], axis=0)
        out_img[ind_c] = class_avg_int
    return out_img
#img = cv2.imread('../Datasets/house.PNG').astype(np.uint8)

#classes_img, k_1 =em_image_seperation(img, 5)
#classes_img_sp, k_2 =em_image_seperation_with_spatial_info(img, 20)

#disp_img_1 = get_color_image_from_classes(k_1, classes_img, img)
#disp_img_2 = get_color_image_from_classes(k_2, classes_img_sp, img)

#cv2.imshow("img", img.astype(np.uint8))
#cv2.imshow("disp_img_1", disp_img_1.astype(np.uint8))
#cv2.imshow("disp_img_2", disp_img_2.astype(np.uint8))
#cv2.waitKey(0)