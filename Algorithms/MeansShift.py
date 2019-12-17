import numpy as np
import cv2
import numpy.lib.recfunctions as rfn
from scipy.ndimage.measurements import label

class Parameters:
    def __init__(self, sigma, parameter_type=0):
        """
        Get parameters for Segmentation
        :param sigma: global sigma from image in feature space
        :param parameter_type: 0 for Undersegmentation, 1 for Oversegmentation else Quantization
        """
        if parameter_type == -1:
            self.r = 0.8*sigma
            self.N_min = 300
            self.N_con = 50
        elif parameter_type == 0:
            self.r = 0.4*sigma
            self.N_min = 400
            self.N_con = 50
        elif parameter_type == 1:
            self.r = 0.3*sigma
            self.N_min = 100
            self.N_con = 10
        else:
            self.r = 0.2*sigma
            self.N_min = 50
            self.N_con = 0
        self.r_sq = self.r*self.r
        expanded_r = self.r*(2.0**(1/3.0))
        self.expanded_r_sq = expanded_r*expanded_r


class Cluster:
    def __init__(self, mode, image_coords, feature_space_pixels):
        self.mode = mode
        self.image_coords = image_coords
        self.feature_space_pixels = feature_space_pixels

    def get_color_mode(self):
        return np.mean(self.feature_space_pixels, axis=0)

    def get_max_connected_components(self, imgshape):
        img_map = np.zeros(imgshape, dtype=np.int)
        img_map[self.image_coords['y_coord'], self.image_coords['x_coord']] = 1
        labeled, ncomponents = label(img_map, np.ones((3, 3), dtype=np.int))
        if ncomponents < 1:
            return 0
        return max([(labeled == i).sum() for i in range(1, ncomponents+1)])

    def remove_all_small_components(self, imgshape, N_con):
        img_map = np.zeros(imgshape, dtype=np.int)
        img_map[self.image_coords['y_coord'], self.image_coords['x_coord']] = 1
        labeled, ncomponents = label(img_map, np.ones((3, 3), dtype=np.int))

        to_be_removed_x = np.empty((0,), dtype=[('x_coord', '<u4')])
        to_be_removed_y = np.empty((0,), dtype=[('y_coord', '<u4')])

        for i in range(1, ncomponents + 1):
            region = labeled == i
            if region.sum() < N_con:
                where_region = np.where(region)
                to_be_removed_x = np.append(to_be_removed_x, where_region[1].astype([('x_coord', 'u4')]))
                to_be_removed_y = np.append(to_be_removed_y, where_region[0].astype([('y_coord', 'u4')]))

        to_be_removed = rfn.merge_arrays((to_be_removed_x, to_be_removed_y), flatten=True, usemask=False)
        to_be_removed_ind = np.where(np.isin(self.image_coords, to_be_removed))

        ret1 = self.image_coords[to_be_removed_ind]
        ret2 = self.feature_space_pixels[to_be_removed_ind]

        self.image_coords = np.delete(self.image_coords, to_be_removed_ind)
        self.feature_space_pixels = np.delete(self.feature_space_pixels, to_be_removed_ind)
        return ret1, ret2


def fast_isotropic_multivariate_gauss(neg_half_sigma, dists_squared_from_mean):
    hold_out = np.exp(neg_half_sigma*dists_squared_from_mean)
    if np.allclose(hold_out, 0):
        return np.ones(hold_out.shape)
    return hold_out


def get_image_indicies(img):
    indx, indy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    return rfn.merge_arrays((indy.astype([('y_coord', 'u4')]), indx.astype([('x_coord', 'u4')])),
                            flatten=True, usemask=False)


def get_coords_in_surrounding_radius(image_co_ordinates, ind_in_radius):
    coords_in_radius = image_co_ordinates[ind_in_radius]
    coords_in_radius_1 = coords_in_radius.copy()
    coords_in_radius_1['y_coord'] += 1
    coords_in_radius_2 = coords_in_radius_1.copy()
    coords_in_radius_2['x_coord'] += 1
    coords_in_radius_3 = coords_in_radius_1.copy()
    coords_in_radius_3['x_coord'] -= 1

    coords_in_radius_4 = coords_in_radius.copy()
    coords_in_radius_4['y_coord'] -= 1
    coords_in_radius_5 = coords_in_radius_4.copy()
    coords_in_radius_5['x_coord'] += 1
    coords_in_radius_6 = coords_in_radius_4.copy()
    coords_in_radius_6['x_coord'] -= 1

    coords_in_radius_7 = coords_in_radius.copy()
    coords_in_radius_7['x_coord'] -= 1
    coords_in_radius_8 = coords_in_radius.copy()
    coords_in_radius_8['x_coord'] += 1
    coords_in_radius_surrounding = np.hstack((coords_in_radius,
                                              coords_in_radius_1,
                                              coords_in_radius_2,
                                              coords_in_radius_3,
                                              coords_in_radius_4,
                                              coords_in_radius_5,
                                              coords_in_radius_6,
                                              coords_in_radius_7,
                                              coords_in_radius_8))
    return np.where(np.isin(image_co_ordinates, coords_in_radius_surrounding))


def get_close_point(test_point, point_list, test_radius_squared, stop_criteria, neg_half_sigma):
    delta = stop_criteria + 1

    while not delta < stop_criteria:
        square_dists = np.sum(np.square(point_list - test_point), axis=1)
        ind_in_radius = np.where(square_dists <= test_radius_squared)

        if len(ind_in_radius[0]) == 0:
            return test_point, ind_in_radius

        points_in_radius = point_list[ind_in_radius]

        weights = fast_isotropic_multivariate_gauss(neg_half_sigma, square_dists[ind_in_radius])

        new_test_point = np.average(points_in_radius, axis=0, weights=weights)

        delta = np.sum(np.square(new_test_point-test_point))
        test_point = new_test_point
        #print(delta)
        #print(test_point)
    return test_point, ind_in_radius


def mean_shift_alg(img, param_mode=0):
    mean_img = cv2.boxFilter(img, cv2.CV_8U, (3, 3))
    feature_space_img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

    image_co_ordinates = get_image_indicies(img)
    flat_feature_space_img = feature_space_img.reshape((-1, feature_space_img.shape[-1]))
    flat_feature_space_mean_img = cv2.cvtColor(mean_img, cv2.COLOR_RGB2Luv).reshape((-1, feature_space_img.shape[-1]))
    flat_feature_space_img_2 = flat_feature_space_img.copy()
    image_co_ordinates_2 = image_co_ordinates.copy()

    sigma = np.std(feature_space_img.flatten())
    params = Parameters(sigma, param_mode)
    M = 200
    clusters = []

    while True:
        #print(len(flat_feature_space_mean_img))
        if len(flat_feature_space_mean_img) <= 0:
            break
        feature_space_candidate_pixels = flat_feature_space_mean_img[np.random.randint(0, len(flat_feature_space_mean_img), M)]

        dist_from_center = np.sum(np.square(flat_feature_space_img[:, None]-feature_space_candidate_pixels), axis=2)

        num_in_each_window = np.sum(dist_from_center <= params.r_sq, axis=0)
        ind_most_in_window = np.argmax(num_in_each_window)

        print(num_in_each_window[ind_most_in_window])
        if num_in_each_window[ind_most_in_window] < params.N_min:
            break

        mode, ind_in_radius = get_close_point(feature_space_candidate_pixels[ind_most_in_window],
                                              flat_feature_space_img, params.r_sq, 0.01, -1)

        new_ind_in_radius = get_coords_in_surrounding_radius(image_co_ordinates, ind_in_radius)

        clusters.append(Cluster(mode, image_co_ordinates[ind_in_radius], flat_feature_space_img[ind_in_radius]))

        image_co_ordinates = np.delete(image_co_ordinates, new_ind_in_radius[0], axis=0)
        flat_feature_space_img = np.delete(flat_feature_space_img, new_ind_in_radius[0], axis=0)
        flat_feature_space_mean_img = np.delete(flat_feature_space_mean_img, new_ind_in_radius[0], axis=0)

        num_ind_in_radius = len(ind_in_radius[0])
        #print(num_ind_in_radius)

    accepted_clusters = []
    for cluster in clusters:
        if cluster.get_max_connected_components(img.shape[:-1]) > params.N_min:
            accepted_clusters.append(cluster)
        # else:
        #    image_co_ordinates = np.append(image_co_ordinates, cluster.image_coords, axis=0)
        #    flat_feature_space_img = np.append(flat_feature_space_img, cluster.feature_space_pixels, axis=0)

    clusters = accepted_clusters


    flat_feature_space_img = flat_feature_space_img_2
    image_co_ordinates = image_co_ordinates_2
    cluster.image_coords = np.empty((0,), dtype=image_co_ordinates.dtype)
    cluster.feature_space_pixels = np.empty((0,flat_feature_space_img.shape[-1]), dtype=flat_feature_space_img.dtype)
    for cluster in clusters:
        square_dists = np.sum(np.square(flat_feature_space_img - cluster.mode), axis=1)
        ind_in_radius = np.where(square_dists <= params.expanded_r_sq)

        cluster.image_coords = np.append(cluster.image_coords, image_co_ordinates[ind_in_radius], axis=0)
        cluster.feature_space_pixels = np.append(cluster.feature_space_pixels,
                                                 flat_feature_space_img[ind_in_radius], axis=0)

        image_co_ordinates = np.delete(image_co_ordinates, ind_in_radius[0], axis=0)
        flat_feature_space_img = np.delete(flat_feature_space_img, ind_in_radius[0], axis=0)

    modes = []
    for cluster in clusters:
        square_dists = np.sum(np.square(flat_feature_space_img - cluster.mode), axis=1)
        ind_in_radius = np.where(square_dists <= params.expanded_r_sq)

        cluster.image_coords = np.append(cluster.image_coords, image_co_ordinates[ind_in_radius], axis=0)
        cluster.feature_space_pixels = np.append(cluster.feature_space_pixels,
                                                 flat_feature_space_img[ind_in_radius], axis=0)

        image_co_ordinates = np.delete(image_co_ordinates, ind_in_radius[0], axis=0)
        flat_feature_space_img = np.delete(flat_feature_space_img, ind_in_radius[0], axis=0)

        modes.append(cluster.mode)

    for i, feature_pixel in enumerate(flat_feature_space_img):
        square_dists = np.sum(np.square(feature_pixel - modes), axis=1)
        min_ind:int = np.argmin(square_dists)
        clusters[min_ind].feature_space_pixels = np.append(clusters[min_ind].feature_space_pixels,
                                                           [feature_pixel], axis=0)
        clusters[min_ind].image_coords = np.append(clusters[min_ind].image_coords,
                                                   image_co_ordinates[i])


    new_img = np.zeros(img.shape)

    for cluster in clusters:
        new_img[cluster.image_coords['y_coord'], cluster.image_coords['x_coord']] = cluster.mode

    return cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_Luv2BGR)

"""
img = cv2.imread('../Datasets/BSDS300/images/train/374067.jpg')
img = cv2.imread('../Datasets/BSDS300/images/train/176035.jpg')
img = cv2.imread('../Datasets/BSDS300/images/train/169012.jpg')

#img = cv2.imread('../Datasets/BSDS300/images/test/37073.jpg')
#img = cv2.imread('../Datasets/house.PNG')

cv2.imshow("img", img)
cv2.imshow("new_img", mean_shift_alg(cv2.imread('../Datasets/BSDS300/images/train/176035.jpg')))
cv2.waitKey(1000)
cv2.imshow("new_img", mean_shift_alg(cv2.imread('../Datasets/BSDS300/images/train/374067.jpg')))
cv2.waitKey(1000)
cv2.imshow("new_img", mean_shift_alg(cv2.imread('../Datasets/BSDS300/images/train/169012.jpg')))
cv2.waitKey(1000)
cv2.imshow("new_img", mean_shift_alg(cv2.imread('../Datasets/BSDS300/images/train/181079.jpg')))
cv2.waitKey(1000)
cv2.imshow("new_img", mean_shift_alg(cv2.imread('../Datasets/BSDS300/images/test/37073.jpg')))
cv2.waitKey(1000000)
"""

