from Algorithms.NormalisedCuts import normalised_cuts
from Algorithms.ExpectationMaximiser import em_image_seperation, em_image_seperation_with_spatial_info, get_color_image_from_classes
from Algorithms.MeansShift import mean_shift_alg
from os import listdir
from os.path import isfile, isdir
from os import makedirs
from os.path import dirname
from matplotlib import pyplot as plt
import cv2
import numpy as np


mean_shift_save_path = "out/mean_shift/"
nc_save_path = "out/nc/"
em_save_path = "out/em/"
em_spat_save_path = "out/em_spat/"
orig_save_path = "out/orig/"

fig_to_save = None
fig_name = None


def get_zero_padded_fig_and_ax(figure_name, figsize=[6, 6]):
    global fig_to_save, fig_name
    fig_name= figure_name
    fig_to_save = plt.figure(0, figsize=figsize)
    ax = fig_to_save.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    return fig_to_save, ax


def save_plot(save_dir, save_format=".pdf"):
    makedirs(dirname(save_dir), exist_ok=True)
    fig_to_save.savefig(
        save_dir + fig_name + save_format, dpi=500, bbox_inches="tight", pad_inches=0)


files_to_process = []
file_names_to_process = []
if not isdir("../Datasets/BSDS300/images/test"):
    print("../Datasets/BSDS300/images/test does not exist. Skipping this directory for now.")
else:
    files_to_process += ["../Datasets/BSDS300/images/test/"+file for file in listdir("../Datasets/BSDS300/images/test")
                         if isfile("../Datasets/BSDS300/images/test/"+file)]
    file_names_to_process += [file for file in listdir("../Datasets/BSDS300/images/test")
                              if isfile("../Datasets/BSDS300/images/test/"+file)]
if not isdir("../Datasets/BSDS300/images/test"):
    print("../Datasets/BSDS300/images/test does not exist. Skipping this directory for now.")
else:
    files_to_process += ["../Datasets/BSDS300/images/test/"+file for file in listdir("../Datasets/BSDS300/images/test")
                         if isfile("../Datasets/BSDS300/images/test/"+file)]
    file_names_to_process += [file for file in listdir("../Datasets/BSDS300/images/train")
                              if isfile("../Datasets/BSDS300/images/train/"+file)]

for i, file in enumerate(files_to_process):
    img = cv2.imread(file)
    _, ax = get_zero_padded_fig_and_ax(str(i), figsize=[6, 6])
    ax.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    save_plot(orig_save_path)

    ms_img = mean_shift_alg(img)
    _, ax = get_zero_padded_fig_and_ax(str(i), figsize=[6, 6])
    ax.imshow(cv2.cvtColor(ms_img.astype(np.uint8), cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    save_plot(mean_shift_save_path)

    classes_img, k_1 = em_image_seperation(img, 5)
    disp_img = get_color_image_from_classes(k_1, classes_img, img)
    _, ax = get_zero_padded_fig_and_ax(str(i), figsize=[6, 6])
    ax.imshow(cv2.cvtColor(disp_img.astype(np.uint8), cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    save_plot(em_save_path)

    classes_img, k_1 = em_image_seperation_with_spatial_info(img, 20)
    disp_img = get_color_image_from_classes(k_1, classes_img, img)
    _, ax = get_zero_padded_fig_and_ax(str(i), figsize=[6, 6])
    ax.imshow(cv2.cvtColor(disp_img.astype(np.uint8), cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    save_plot(em_spat_save_path)

    classes_img, num_classes = normalised_cuts(img, thresh=0.5, r_sq=5**2, sigma_x_sq=2**2, sigma_i_sq=0.1**2)
    disp_img = get_color_image_from_classes(num_classes, classes_img, img)
    _, ax = get_zero_padded_fig_and_ax(str(i), figsize=[6, 6])
    ax.imshow(cv2.cvtColor(disp_img.astype(np.uint8), cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    save_plot(nc_save_path)
