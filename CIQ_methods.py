# Modules calling
import cv2
import pandas as pd
import numpy as np
import os
import time
from deap import base, creator, tools
import random
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim, compare_psnr, compare_mse


def KMeans_Clastering(input_img, org_img, C, step=1):
    # define criteria, number of clusters(K) and apply kmeans()
    # Set flags (Just to avoid line break in the code)
    input_vec = np.reshape(input_img, (input_img.shape[0] * input_img.shape[1], input_img.shape[2]))
    org_vec = np.reshape(org_img, (org_img.shape[0] * org_img.shape[1], org_img.shape[2]))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    ret, label, center = cv2.kmeans(input_vec[::step], C, None, criteria, 10, flags)

    # represet out_vec based on center
    err = np.zeros((C, org_vec.shape[0], 3), dtype=np.float32)  # tem[K][65536][3]
    for i in range(C):
        err[i] = (org_vec - center[i]) ** 2

    dist = np.sum(err, axis=2)
    dist = np.sqrt(dist)
    dis = dist ** 2
    cln = np.argmin(dis, axis=0)
    cln = cln.reshape((-1, 1))

    # convert back into uint8, and make original image
    center = np.uint8(center)
    cln = np.uint8(cln)
    out_vec = center[cln.flatten()]
    output_img = np.reshape(out_vec, org_img.shape)
    return output_img, center


def get_MaxMinErrorLocalArea(img1, img2, window_size, without_point):
    max_position = [0, 0]
    min_position = [0, 0]
    max_err = 0
    min_err = 1e15
    # 局所窓画像の左上の座標を求める
    y_list = [n for n in range(0, img1.shape[0], window_size)]
    x_list = [n for n in range(0, img1.shape[1], window_size)]

    for y in y_list:
        flag = False
        for x in x_list:
            for point in without_point:
                if point[0] == y and point[1] == x:
                    flag = True
                    break

            local_error_area = np.abs(img1[y: y + window_size, x: x + window_size] - \
                                      img2[y: y + window_size, x: x + window_size])
            err = np.sum(local_error_area)

            if max_err < err:
                max_position = [y, x]
                max_err = err

            if not flag:
                if min_err > err:
                    min_position = [y, x]
                    min_err = err

    return np.array(max_position), np.array(min_position), max_err, min_err


def get_MaxMinErrorLocalArea_byResize(clustered_img, org_img, window_size):
    err = np.abs(clustered_img - org_img)
    resized_err = cv2.resize(err,
                             (int(clustered_img.shape[0] / window_size), int(clustered_img.shape[1] / window_size)),
                             interpolation=cv2.INTER_AREA)
    resized_sum_err = np.sum(resized_err, axis=2)

    def calc_MaxMinError(without_position):
        max_index = np.argmax(resized_sum_err)
        linier_sum_err = np.reshape(resized_sum_err, resized_sum_err.shape[0] * resized_sum_err.shape[1])
        sorted_index = np.argsort(linier_sum_err)
        min_index = sorted_index[0]
        for index in sorted_index:
            y = int(index / resized_err.shape[0]) * window_size
            x = int(index % resized_err.shape[0]) * window_size
            for position in without_position:
                if y == position[0] and x == position[1]:
                    break
            else:
                min_index = index
                break

        max_position = [int(max_index / resized_err.shape[0]) * window_size,
                        int(max_index % resized_err.shape[0]) * window_size]
        min_position = [int(min_index / resized_err.shape[0]) * window_size,
                        int(min_index % resized_err.shape[0]) * window_size]

        return max_position, min_position

    def replace_max2min(palette_img, min_position, max_position, window_size):
        img = palette_img.copy()
        img[min_position[0]: min_position[0] + window_size, min_position[1]: min_position[1] + window_size] = \
            org_img[max_position[0]: max_position[0] + window_size,
            max_position[1]: max_position[1] + window_size].copy()

        return img

    return resized_sum_err, calc_MaxMinError, replace_max2min


def replace_max2minErrorArea(min_position, max_position, org_img, window_size):
    img = org_img.copy()
    img[min_position[0]: min_position[0] + window_size, min_position[1]: min_position[1] + window_size] = \
        org_img[max_position[0]: max_position[0] + window_size, max_position[1]: max_position[1] + window_size].copy()

    return img


def Test_Iterative_KMeans(img, window_size, times, K=16, dir=''):
    if not os.path.isdir(dir):
        print("make {} dir".format(dir))
        os.mkdir(dir)
    save_dir = "class_{}_window_{}".format(K, window_size)
    save_dir = os.path.join(dir, save_dir)
    if not os.path.isdir(save_dir):
        print("make {} dir".format(save_dir))
        os.mkdir(save_dir)

    palette_img = img.copy()
    without_point = []
    # save the number of iterations and each MSE, suggestion Distances
    x = []
    max_errs = []
    min_errs = []
    mse = []

    # 5, 10, 50, 75, 100, 200
    weights = [5, 10, 50, 75, 100, 200]
    sug_errors = np.zeros((len(weights), times))

    for n in range(times):
        window_errors = []
        weighted_window_errors = []
        output_img, centers = KMeans_Clastering(palette_img, img, K)
        max_position, min_position, max_err, min_err = \
            get_MaxMinErrorLocalArea(output_img, img, window_size, without_point)

        max_errs.append(max_err)
        min_errs.append(min_err)
        without_point.append(min_position)

        tmp_err_img = np.abs(output_img - img)
        err_img = np.zeros(img.shape)
        for c in range(3):
            err_img[:, :, 0] += tmp_err_img[:, :, c]
            err_img[:, :, 1] += tmp_err_img[:, :, c]
            err_img[:, :, 2] += tmp_err_img[:, :, c]

        err_img /= 3.0

        palette_img = replace_max2minErrorArea(min_position, max_position, palette_img, window_size)

        output_img_name = os.path.join(save_dir, "{}times_output.bmp".format(n))
        palette_img_name = os.path.join(save_dir, "{}times_palette.bmp".format(n))
        err_img_name = os.path.join(save_dir, "{}times_error.bmp".format(n))

        cv2.imwrite(output_img_name, output_img)
        cv2.imwrite(palette_img_name, palette_img)
        cv2.imwrite(err_img_name, err_img)
        print('save {}time palette and output imgs'.format(n))

        x.append(n)
        mse.append(get_mse(output_img, img))

        for num in range(len(weights)):
            sug_errors[num, n], sorted_window, weighted_window = get_sug1_error(output_img, img, window_size, 1,
                                                                                weights[num])
            if num == 0:
                window_errors.append(sorted_window)
            weighted_window_errors.append(weighted_window)

        fig = plt.figure()
        x_index = np.arange(0, len(window_errors[0]))
        sorted_ax = fig.add_subplot(1, 1, 1)
        sorted_ax.plot(x_index, window_errors[0], label='normal sort')
        plt.tight_layout()
        sorting_file_name = os.path.join(save_dir, 'WindowError_{}times.bmp'.format(n))
        plt.savefig(sorting_file_name)
        plt.close(fig)

        fig = plt.figure()
        for num in range(len(weights)):
            # 重みが6種類の場合のみ対応してる
            weighted_ax = fig.add_subplot(3, 2, num + 1)
            weighted_ax.plot(x_index, weighted_window_errors[num], label='weighted sort w = {}'.format(weights[num]))
            weighted_ax.set_title('w = {}'.format(weights[num]))

        plt.tight_layout()
        sorting_file_name = os.path.join(save_dir, 'WeightedWindowError_{}times.bmp'.format(n))
        plt.savefig(sorting_file_name)
        plt.close(fig)

    fig = plt.figure()
    n_window = 3

    tmp_index = ['MSE', 'max_errs', 'min_errs']
    mse_position = fig.add_subplot(n_window, n_window, 1)
    mse_position.plot(x, mse, label='MSE')
    mse_position.set_title(tmp_index[0])

    min_pos = fig.add_subplot(n_window, n_window, 2)
    min_pos.plot(x, min_errs, label='MIN_ERRS')
    min_pos.set_title(tmp_index[1])

    max_pos = fig.add_subplot(n_window, n_window, 3)
    max_pos.plot(x, max_errs, label='MAX_ERRS')
    max_pos.set_title(tmp_index[2])

    for n in range(len(weights)):
        position = fig.add_subplot(n_window, n_window, n + 4)
        position.plot(x, sug_errors[n], label='a = {}'.format(weights[n]))
        position.set_title('w = {}'.format(weights[n]))

    fig_name = os.path.join(save_dir, 'Error.bmp')
    plt.tight_layout()
    plt.savefig(fig_name)
    print('save {} img'.format(fig_name))

    arr = np.array([mse, max_errs, min_errs])
    arr = np.vstack((arr, sug_errors))

    df = pd.DataFrame(arr)
    for w in weights:
        tmp_index.append('a = {}'.format(w))
    df.index = tmp_index

    csv_name = os.path.join(save_dir, 'MSE_and_SugDist.csv')
    df.to_csv(csv_name)
    print('save {} csv'.format(csv_name))
    plt.close(fig)


def Iterative_KMeans(img, window_size, times, K=16):
    palette_img = img.copy()
    without_point = []
    output_img = img.copy()
    for n in range(times):
        output_img, centers = KMeans_Clastering(palette_img, img, K)
        max_position, min_position = get_MaxMinErrorLocalArea(output_img, img, window_size, without_point)
        without_point.append(min_position)

        tmp_err_img = np.abs(output_img - img)
        err_img = np.zeros(img.shape)
        for c in range(3):
            err_img[:, :, 0] += tmp_err_img[:, :, c]
            err_img[:, :, 1] += tmp_err_img[:, :, c]
            err_img[:, :, 2] += tmp_err_img[:, :, c]

        err_img /= 3.0

        palette_img = replace_max2minErrorArea(min_position, max_position, palette_img, window_size)

    return output_img


def Iterative_KMeans_byResizing(img, window_size, times, K=16):
    palette_img = img.copy()
    without_position = []
    output_img = img.copy()
    for n in range(times):
        output_img, centers = KMeans_Clastering(palette_img, img, K)
        resized_err, calc_MaxMin, replace = get_MaxMinErrorLocalArea_byResize(output_img, img, window_size)
        max_position, min_position = calc_MaxMin(without_position)
        without_position.append(min_position)
        palette_img = replace(palette_img, min_position, max_position, window_size)

    return output_img


def Test_Iterative_KMeans_byResizing(img, window_size, times, K=16, dir=''):
    save_dir = "resized_class_{}_window_{}".format(K, window_size)
    save_dir = os.path.join(dir, save_dir)
    if not os.path.isdir(save_dir):
        print("make {} dir".format(save_dir))
        os.mkdir(save_dir)

    palette_img = img.copy()
    without_position = []

    # save the number of iterations and each MSE, suggestion Distances
    x = []
    mse = []
    sug5 = []
    sug10 = []
    sug50 = []
    sug75 = []
    sug100 = []
    for n in range(times):
        output_img, centers = KMeans_Clastering(palette_img, img, K)
        resized_err, calc_MaxMin, replace = get_MaxMinErrorLocalArea_byResize(output_img, img, window_size)
        max_position, min_position = calc_MaxMin(without_position)
        without_position.append(min_position)
        palette_img = replace(palette_img, min_position, max_position, window_size)

        output_img_name = os.path.join(save_dir, "{}times_output.bmp".format(n))
        palette_img_name = os.path.join(save_dir, "{}times_palette.bmp".format(n))
        err_img_name = os.path.join(save_dir, "{}times_error.bmp".format(n))

        cv2.imwrite(output_img_name, output_img)
        cv2.imwrite(palette_img_name, palette_img)
        cv2.imwrite(err_img_name, resized_err)
        print('save {}time palette and output imgs'.format(n + 1))

        x.append(n)
        mse.append(get_mse(output_img, img))
        sug5.append(get_sug1_error(output_img, img, window_size, 1, 5))
        sug10.append(get_sug1_error(output_img, img, window_size, 1, 10))
        sug50.append(get_sug1_error(output_img, img, window_size, 1, 50))
        sug75.append(get_sug1_error(output_img, img, window_size, 1, 75))
        sug100.append(get_sug1_error(output_img, img, window_size, 1, 100))

    fig = plt.figure()
    mse_position = fig.add_subplot(6, 1, 1)
    sug5_position = fig.add_subplot(6, 1, 2)
    sug10_position = fig.add_subplot(6, 1, 3)
    sug50_position = fig.add_subplot(6, 2, 1)
    sug75_position = fig.add_subplot(6, 2, 2)
    sug100_position = fig.add_subplot(6, 2, 3)

    mse_position.scatter(x, mse, label='MSE')
    sug5_position.scatter(x, sug5, label='a = 5')
    sug10_position.scatter(x, sug10, label='a = 10')
    sug50_position.scatter(x, sug50, label='a = 50')
    sug75_position.scatter(x, sug75, label='a = 75')
    sug100_position.scatter(x, sug100, label='a = 100')
    for pos in [mse_position, sug5_position, sug10_position, sug50_position, sug75_position, sug100_position]:
        pos.set_xlabel('Iterations')
        pos.set_ylabel('Error')

    fig_name = os.path.join(save_dir, 'Error.bmp')
    plt.savefig(fig_name)
    print('save {} img'.format(fig_name))

    arr = np.array([mse, sug5, sug10, sug50, sug75, sug100])
    df = pd.DataFrame(arr)
    df.reindex(['MSE', 'a = 5', 'a = 10', 'a = 50', 'a = 75', 'a = 100'])
    csv_name = os.path.join(save_dir, 'MSE_and_SugDist.csv')
    df.to_csv(csv_name)
    print('save {} csv'.format(csv_name))


def ISRandCIQ(img, window_size, times, R=2, K=16):
    # image enleargement
    sr_img = cv2.resize(img, fx=R, fy=R, interpolation=cv2.INTER_CUBIC)

    palette_img = img.copy()
    output_img = img.copy()
    without_point = []
    for n in range(times):
        output_img, centers = KMeans_Clastering(palette_img, sr_img, K)
        err_img = np.abs(output_img - sr_img)
        rd_err_img = cv2.resize(err_img, fx=(1.0 / R), fy=(1.0 / R), interpolation=cv2.INTER_AREA)

        linered_rd = np.reshape(rd_err_img, (rd_err_img.shape[0] * rd_err_img.shape[1]))
        max_index = np.argmax(linered_rd)
        sort_index = np.argsort(linered_rd)
        min_position = [0, 0]
        max_position = [int(max_index / rd_err_img.shape[0]), int(max_index % rd_err_img.shape[1])]

        for index in sort_index:
            x = int(index % rd_err_img.shape[0])
            y = int(index / rd_err_img.shape[0])
            for position in without_point:
                if position[0] == y and position[1] == x:
                    break
            else:
                min_position = [y, x]
                break

        without_point.append(min_position)
        palette_img = replace_max2minErrorArea(min_position, max_position, palette_img, window_size)

    return output_img


def draw_MaxMin(error_img):
    modified = error_img.copy()
    lineared = np.reshape(error_img, error_img.shape[0] * error_img.shape[1])
    max_index = np.argmax(lineared)
    min_index = np.argmin(lineared)

    max_y, max_x = int(max_index / error_img.shape[0]), int(max_index % error_img.shape[1])
    min_y, min_x = int(min_index / error_img.shape[0]), int(min_index % error_img.shape[1])

    modified[max_y, max_x] = 150
    modified[min_y, min_x] = 255

    return modified


def Test_ISRandCIQ_KMeans(img, window_size, times, img_name, R=2, K=16, sr_num=0):
    interpolation_name = ['NNI', 'BLI', 'BCI', 'LZI']
    save_dir = "SR_class_{}_window_{}_Ratio_{}_KMeans".format(K, window_size, R)

    if not os.path.isdir(img_name):
        print("make {} dir".format(img_name))
        os.mkdir(img_name)
    save_dir = os.path.join(img_name, save_dir)

    if not os.path.isdir(save_dir):
        print("make {} dir".format(save_dir))
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, interpolation_name[sr_num])
    if not os.path.isdir(save_dir):
        print("make {} dir".format(save_dir))
        os.mkdir(save_dir)

    interpolation = cv2.INTER_NEAREST
    if sr_num == 1:
        interpolation = cv2.INTER_LINEAR
    elif sr_num == 2:
        interpolation = cv2.INTER_CUBIC
    elif sr_num == 3:
        interpolation = cv2.INTER_LANCZOS4

    zoom_img = cv2.resize(img, (int(img.shape[0] / R), int(img.shape[1] / R)), interpolation=cv2.INTER_AREA)
    sr_img = cv2.resize(zoom_img, (int(img.shape[0]), int(img.shape[1])), interpolation=interpolation)
    sr_img_filename = os.path.join(save_dir, img_name + '.bmp')
    cv2.imwrite(sr_img_filename, sr_img)
    print('saved sr img {}'.format(sr_img_filename))

    palette_img = img.copy()
    output_img = img.copy()

    weights = [1, 2, 10, 20, 50]
    sug3_weights = [1, 5, 10, 20, 50]
    sug4_weights = [5, 10, 20]
    sug3_bias = [0.1, 0.2, 0.5]
    sug_errors = np.zeros((len(weights), times + 1))
    sug3_errors = np.zeros((int(len(sug3_weights) * len(sug3_bias)), times + 1))
    sug4_errors = np.zeros((len(sug4_weights), times + 1))
    without_point = []
    window_errors = []
    weighted_window_errors = []
    mses = []
    psnrs = []
    ssims = []
    mses.append(compare_mse(sr_img, img))
    psnrs.append(compare_psnr(sr_img, img, data_range=255))
    ssims.append(compare_ssim(sr_img, img, data_range=255.0, multichannel=True))
    for num in range(len(weights)):
        sug_errors[num, 0], sorted_window, weighted_window = get_sug1_error(sr_img, img, window_size, 1,
                                                                            weights[num])
        if num == 0:
            window_errors.append(sorted_window)
        weighted_window_errors.append(weighted_window)

    for n in range(times):
        output_img, centers = KMeans_Clastering(palette_img, sr_img, K)

        err_img = np.abs(output_img - sr_img)

        rd_err_img = cv2.resize(err_img, (int(img.shape[0]), int(img.shape[1])), interpolation=cv2.INTER_AREA)

        # 1 / window_sizeに縮小することで、局所領域の計算の省略
        resized_rd_img = cv2.resize(rd_err_img,
                                    (int(rd_err_img.shape[0] / window_size), int(rd_err_img.shape[1] / window_size)),
                                    interpolation=cv2.INTER_AREA)
        gray_err_img = cv2.cvtColor(resized_rd_img, cv2.COLOR_BGR2GRAY)
        gray_err_img = draw_MaxMin(gray_err_img)
        resized_rd_img = np.sum(resized_rd_img, axis=2)

        linered_rd = np.reshape(resized_rd_img, (resized_rd_img.shape[0] * resized_rd_img.shape[1]))
        max_index = np.argmax(linered_rd)
        sort_index = np.argsort(linered_rd)
        min_position = [0, 0]

        max_position = [int(int(max_index / resized_rd_img.shape[0]) * window_size),
                        int(int(max_index % resized_rd_img.shape[1]) * window_size)]

        for index in sort_index:
            x = int(int(index % resized_rd_img.shape[0]) * window_size)
            y = int(int(index / resized_rd_img.shape[0]) * window_size)
            for position in without_point:
                if position[0] == y and position[1] == x:
                    break
            else:
                min_position = [y, x]
                break

        # max_position, min_position = get_MaxMinErrorLocalArea(sr_img, output_img, window_size, without_point)
        # max_position = (max_position / R).astype(np.int8)
        # min_position = (min_position / R).astype(np.int8)

        without_point.append(min_position)
        palette_img = replace_max2minErrorArea(min_position, max_position, palette_img, window_size)
        tmp_mse = compare_mse(output_img, img)
        tmp_psnr = compare_psnr(output_img, img, data_range=255)
        tmp_ssim = compare_ssim(output_img, img, multichannel=True, data_range=255.0)

        # 各距離指標による評価
        for num in range(len(weights)):
            sug_errors[num, n + 1], sorted_window, weighted_window = get_sug1_error(output_img, img, window_size, 1,
                                                                                weights[num])
            if num == 0:
                window_errors.append(sorted_window)
            weighted_window_errors.append(weighted_window)

        for i in range(len(sug3_weights)):
            for j in range(len(sug3_bias)):
                sug3_errors[i * len(sug3_bias) + j, n + 1], _, __ = get_sug1_error(output_img, img, window_size, 3,
                                                                                   sug3_weights[i])
        for i in range(len(sug4_weights)):
            sug4_errors[i, n + 1], _, __ = get_sug1_error(output_img, img, window_size, 3,
                                                                               sug4_weights[i])

        mses.append(tmp_mse)
        psnrs.append(tmp_psnr)
        ssims.append(tmp_ssim)

        output_img_name = os.path.join(save_dir, "{}times_output.bmp".format(n + 1))
        palette_img_name = os.path.join(save_dir, "{}times_palette.bmp".format(n + 1))
        err_img_name = os.path.join(save_dir, "{}times_error.bmp".format(n + 1))
        sum_err_name = os.path.join(save_dir, "{}times_sum_error.bmp".format(n + 1))

        cv2.imwrite(output_img_name, output_img)
        cv2.imwrite(palette_img_name, palette_img)
        cv2.imwrite(err_img_name, gray_err_img)
        cv2.imwrite(sum_err_name, resized_rd_img)
        print('save {}time palette and output imgs'.format(n + 1))

    arr = np.array([mses, psnrs, ssims])
    tmp_index = ['MSE(the first cells are showed as distances between org and resized one)', 'PSNR', 'SSIM']
    tmp_columns = tmp_index.copy()

    for w in weights:
        tmp_columns.append('a = {}'.format(w))
        tmp_index.append('a = {}'.format(w))

    for w in sug3_weights:
        for b in sug3_bias:
            tmp_columns.append('a = {}, b = {}'.format(w, b))
            tmp_index.append('a = {}, b = {}'.format(w, b))
    for w in sug4_weights:
        tmp_columns.append('a = {}'.format(w))
        tmp_index.append('a = {}'.format(w))

    arr = np.concatenate([arr, sug_errors, sug3_errors, sug4_errors], axis=0)

    df = pd.DataFrame(arr)
    min_df = pd.DataFrame()
    for i in range(len(df)):
        if tmp_index[i] == 'PSNR' or tmp_index[i] == 'SSIM':
            # best_val = np.max(df.iloc[i, 1])
            selected_index = np.argmax(df.iloc[i, 1:])
        else:
            # best_val = np.min(df.iloc[i, 1])
            selected_index = np.argmin(df.iloc[i, 1:])

        selected_data = df.iloc[:, selected_index]
        df_best_val = pd.DataFrame([selected_index])
        selected_data = selected_data.append(df_best_val)
        min_df = min_df.append([selected_data.T])

    df.index = tmp_index
    tmp_columns.append('best number')
    min_df.index = tmp_index
    min_df.columns = tmp_columns

    csv_name = os.path.join(save_dir, 'measurement.csv')
    min_csv_name = os.path.join(save_dir, 'bests.csv')
    df.to_csv(csv_name)
    min_df.to_csv(min_csv_name)
    print('save {} csv'.format(csv_name))

    return output_img


def Median_Cut(img, K, step=1):
    src_vec = img.reshape((-1, 3))
    src_vec = np.float32(src_vec)
    height = img.shape[0]
    width = img.shape[1]
    count_color = height * width

    in_vec = np.ones((count_color, 4))
    in_vec[:, :3] = src_vec
    in_vec_idx = np.arange(count_color)

    num_color = K
    for j in range(num_color - 1):
        num_label = j + 1
        in_class_dif = np.zeros((num_label, 3), dtype=np.float32)

        if (num_label > 1):
            for i in range(num_label):
                in_class = in_vec[in_vec[:, 3] == (i + 1), :3]
                in_class_max = np.max(in_class, axis=0)
                in_class_min = np.min(in_class, axis=0)
                in_class_dif[i] = in_class_max - in_class_min
            dif_max = np.max(in_class_dif, axis=0)
            max_col = np.max(in_class_dif, axis=1)
            label = np.argmax(max_col) + 1
            comp_num = np.argmax(dif_max)

        elif (num_label == 1):
            label = 1
            in_class = in_vec[:, :3]
            in_class_max = np.max(in_class, axis=0)
            in_class_min = np.min(in_class, axis=0)
            in_class_dif = in_class_max - in_class_min
            dif_max = np.max(in_class_dif)
            comp_num = np.argmax(in_class_dif)
        else:
            pass

        in_class = in_vec[in_vec[:, 3] == label, :3]
        in_extra = in_vec_idx[in_vec[:, 3] == label]
        center_idx = int(np.ceil(in_class.shape[0] / 2))

        sort_class = np.sort(in_class, axis=0)
        sort_class_idx = np.argsort(in_class, axis=0)

        new_label = num_label + 1

        in_vec[in_extra[sort_class_idx[(center_idx + 1):sort_class_idx.shape[0], comp_num]], 3] = new_label

    num_label = np.max(in_vec[:, 3])
    class_mean = np.zeros((int(num_label), 3), dtype=np.float32)
    for i in range(int(num_label)):
        in_class = in_vec[in_vec[:, 3] == i + 1, :3]
        in_class_size = in_class.shape[0]
        class_mean[i] = np.mean(in_class, axis=0)
        in_vec[in_vec[:, 3] == i + 1, :3] = np.matlib.repmat(class_mean[i], in_class_size, 1)

    norms = np.zeros((num_color, 1))
    out_vec = np.zeros((height * width, 3))
    cube_num = np.zeros((height * width, 1), dtype=np.uint8)
    tem_vec = np.zeros((num_color, height * width, 3), dtype=np.float32)
    dis_vec = np.zeros((height * width, 1), dtype=np.float32)
    for i in range(num_color):
        tem_vec[i] = src_vec - class_mean[i]
    tem_vec = tem_vec ** 2
    dis_vec = np.sum(tem_vec, axis=2)
    dis_vec = np.sqrt(dis_vec)
    dis_vec = dis_vec ** 2
    cube_num = np.argmin(dis_vec, axis=0)

    class_mean = np.uint8(class_mean)
    out_vec = class_mean[cube_num.flatten()]

    mca_img = out_vec.reshape((img.shape))

    return mca_img


def _median_cut(img, K, step=10):
    img = img.astype(np.int32)
    img_vec = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    color_hist = np.zeros((256, 256, 256)) # B, G, R
    clusters = []
    for vec in img_vec[::step]:
        color_hist[vec] += 1

    for n in range(K):
        r_max, r_min = np.max(img[:, :, 2]), np.min(img[:, :, 2])
        g_max, g_min = np.max(img[:, :, 1]), np.min(img[:, :, 1])
        b_max, b_min = np.max(img[:, :, 0]), np.min(img[:, :, 0])
        color_hist = color_hist[b_min: b_max + 1, g_min: g_max + 1, r_min: r_max + 1]

        longest_index = np.argmax(color_hist.shape)

        sum = color_hist.copy()
        for ch in range(3):
            if ch != longest_index:
                sum = np.sum(sum, axis=ch)

        med = np.median(sum, axis=longest_index)
        med_index = np.where(med == sum)[0]

        # separating
        for ch in range(3):
            if ch == longest_index:
                if longest_index == 0:
                    cluster_1 = color_hist[:med_index, :, :]
                    cluster_2 = color_hist[med_index:, :, :]
                elif longest_index == 1:
                    cluster_1 = color_hist[:, :med_index, :]
                    cluster_2 = color_hist[:, med_index:, :]
                elif longest_index == 2:
                    cluster_1 = color_hist[:, :, :med_index]
                    cluster_2 = color_hist[:, :, med_index:]

                if np.sum(cluster_1.shape) > np.sum(cluster_2.shape):
                    clusters.append(cluster_2)

                    if n == (K - 1):
                        clusters.append(cluster_2)
                    else:
                        color_hist = cluster_2

    for cluster in clusters:
        mean = np.mean(cluster)


def Test_ISRandCIQ_MedianCut(img, window_size, times, img_name, R=2, K=16, sr_num=0):
    interpolation_name = ['NNI', 'BLI', 'BCI', 'LZI']
    save_dir = "SR_class_{}_window_{}_Ratio_{}_MedianCut".format(K, window_size, R)

    if not os.path.isdir(img_name):
        print("make {} dir".format(img_name))
        os.mkdir(img_name)
    save_dir = os.path.join(img_name, save_dir)

    if not os.path.isdir(save_dir):
        print("make {} dir".format(save_dir))
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, interpolation_name[sr_num])
    if not os.path.isdir(save_dir):
        print("make {} dir".format(save_dir))
        os.mkdir(save_dir)

    interpolation = cv2.INTER_NEAREST
    if sr_num == 1:
        interpolation = cv2.INTER_LINEAR
    elif sr_num == 2:
        interpolation = cv2.INTER_CUBIC
    elif sr_num == 3:
        interpolation = cv2.INTER_LANCZOS4

    zoom_img = cv2.resize(img, (int(img.shape[1] / R), int(img.shape[0] / R)), interpolation=cv2.INTER_AREA)
    sr_img = cv2.resize(zoom_img, (int(img.shape[1]), int(img.shape[0])), interpolation=interpolation)
    sr_img_filename = os.path.join(save_dir, img_name + '.bmp')
    cv2.imwrite(sr_img_filename, sr_img)
    print('saved sr img {}'.format(sr_img_filename))

    palette_img = img.copy()
    output_img = img.copy()

    weights = [5, 10, 50, 75, 100, 200]
    sug_errors = np.zeros((len(weights), times + 1))
    without_point = []
    window_errors = []
    weighted_window_errors = []
    mses = []
    psnrs = []
    ssims = []
    mses.append(compare_mse(sr_img, img))
    psnrs.append(compare_psnr(sr_img, img, data_range=255))
    ssims.append(compare_ssim(sr_img, img, data_range=255.0, multichannel=True))
    for num in range(len(weights)):
        sug_errors[num, 0], sorted_window, weighted_window = get_sug1_error(sr_img, img, window_size, 1,
                                                                            weights[num])
        if num == 0:
            window_errors.append(sorted_window)
        weighted_window_errors.append(weighted_window)

    for n in range(times):
        # output_img, centers = KMeans_Clastering(palette_img, sr_img, K)
        output_img = _SLFA(img, K)

        err_img = np.abs(output_img - sr_img)

        rd_err_img = cv2.resize(err_img, (int(img.shape[0]), int(img.shape[1])), interpolation=cv2.INTER_AREA)

        # 1 / window_sizeに縮小することで、局所領域の計算の省略
        resized_rd_img = cv2.resize(rd_err_img,
                                    (int(rd_err_img.shape[0] / window_size), int(rd_err_img.shape[1] / window_size)),
                                    interpolation=cv2.INTER_AREA)
        gray_err_img = cv2.cvtColor(resized_rd_img, cv2.COLOR_BGR2GRAY)
        gray_err_img = draw_MaxMin(gray_err_img)
        resized_rd_img = np.sum(resized_rd_img, axis=2)

        linered_rd = np.reshape(resized_rd_img, (resized_rd_img.shape[0] * resized_rd_img.shape[1]))
        max_index = np.argmax(linered_rd)
        sort_index = np.argsort(linered_rd)
        min_position = [0, 0]

        max_position = [int(int(max_index / resized_rd_img.shape[0]) * window_size),
                        int(int(max_index % resized_rd_img.shape[1]) * window_size)]

        for index in sort_index:
            x = int(int(index % resized_rd_img.shape[0]) * window_size)
            y = int(int(index / resized_rd_img.shape[0]) * window_size)
            for position in without_point:
                if position[0] == y and position[1] == x:
                    break
            else:
                min_position = [y, x]
                break

        # max_position, min_position = get_MaxMinErrorLocalArea(sr_img, output_img, window_size, without_point)
        # max_position = (max_position / R).astype(np.int8)
        # min_position = (min_position / R).astype(np.int8)

        without_point.append(min_position)
        palette_img = replace_max2minErrorArea(min_position, max_position, palette_img, window_size)
        tmp_mse = compare_mse(output_img, img)
        tmp_psnr = compare_psnr(output_img, img, data_range=255)
        tmp_ssim = compare_ssim(output_img, img, multichannel=True, data_range=255.0)

        for num in range(len(weights)):
            sug_errors[num, n + 1], sorted_window, weighted_window = get_sug1_error(output_img, img, window_size, 1,
                                                                                weights[num])
            if num == 0:
                window_errors.append(sorted_window)
            weighted_window_errors.append(weighted_window)

        mses.append(tmp_mse)
        psnrs.append(tmp_psnr)
        ssims.append(tmp_ssim)

        output_img_name = os.path.join(save_dir, "{}times_output.bmp".format(n + 1))
        palette_img_name = os.path.join(save_dir, "{}times_palette.bmp".format(n + 1))
        err_img_name = os.path.join(save_dir, "{}times_error.bmp".format(n + 1))
        sum_err_name = os.path.join(save_dir, "{}times_sum_error.bmp".format(n + 1))

        cv2.imwrite(output_img_name, output_img)
        cv2.imwrite(palette_img_name, palette_img)
        cv2.imwrite(err_img_name, gray_err_img)
        cv2.imwrite(sum_err_name, resized_rd_img)
        print('save {}time palette and output imgs'.format(n + 1))

    arr = np.array([mses, psnrs, ssims])
    tmp_index = ['MSE(the first cells are showed as distances between org and resized one)', 'PSNR', 'SSIM']

    arr = np.concatenate([arr, sug_errors], axis=0)

    df = pd.DataFrame(arr)
    for w in weights:
        tmp_index.append('a = {}'.format(w))
    df.index = tmp_index

    csv_name = os.path.join(save_dir, 'measurement.csv')
    df.to_csv(csv_name)
    print('save {} csv'.format(csv_name))

    return output_img


def Shuffled_Leaping_Frogs_Algorithm(cluster, K):
    population = 50
    memplex = 5
    variables = 10
    upperlimit = 255
    lowerlimit = 0
    Dmax = 100
    total = 0
    global_best = 0
    n = int(population / memplex)
    miteration = 8
    fitness = [0 for i in range(population)]
    # generating random frogs
    frogs = [[0 for i in range(variables)] for j in range(population)]
    for i in range(population):
        for j in range(variables):
            frogs[i][j] = (random.random() * upperlimit) + (random.random() * lowerlimit)
            if frogs[i][j] < 0:
                frogs[i][j] = frogs[i][j] + 100

    # calculating the fitness value
    def fitness_fn(frogs):
        global fitness
        for i in range(population):
            total = 0
            for j in range(variables):
                total = total + frogs[i][j] * frogs[i][j]
            fitness[i] = total

    def fit(check):
        total = 0
        for j in range(variables):
            total = total + check[j] * check[j]
        return total

    for z in range(population):
        fitness[z] = fit(frogs[z])

    # arranging in descending order
    for j in range(population - 1):
        for k in range(j + 1, population):
            if fitness[j] < fitness[k]:
                temp = fitness[j]
                fitness[j] = fitness[k]
                fitness[k] = temp
                temp2 = frogs[j]
                frogs[j] = frogs[k]
                frogs[k] = temp2

    # assigning the global best
    global_best = fitness[0]

    # creating the memplexes
    pop = 0
    memplexes = [[[0 for i in range(variables)] for j in range(n)] for k in range(memplex)]

    for i in range(memplex):
        for j in range(n):
            for k in range(variables):
                memplexes[i][j][k] = frogs[pop][k]
            pop += 1

    for i in range(memplex):
        print("memplex", i)
        for j in range(miteration):
            bestfrog = memplexes[i][0]
            worstfrog = memplexes[i][n - 1]
            fw = 0
            fw = fit(worstfrog)
            for t in range(variables):
                di = random.random() * (bestfrog[t] - worstfrog[t])
                if (di < -Dmax):
                    di = Dmax
                if (di > Dmax):
                    di = Dmax
                worstfrog[t] = worstfrog[t] + di
            fn = 0
            fn = fit(worstfrog)
            print("old worst ", fw, " new worst ", fn)
            if (fn < fw):
                bestfrog = frogs[0]
                for u in range(variables):
                    di = random.random() * (bestfrog[u] - worstfrog[u])
                    if (di < -Dmax):
                        di = Dmax
                    if (di > Dmax):
                        di = Dmax
                    worstfrog[u] = worstfrog[u] + di
                fn = fit(worstfrog)
                print("old worst1 ", fw, " new worst1 ", fn)
                if (fn < fw):
                    for v in range(variables):
                        worstfrog[v] = random.random() * upperlimit + random.random() * lowerlimit
                    memplexes[i][n - 1] = worstfrog
                    fn = fit(worstfrog)
    print("old worst2 ", fw, " new worst2 ", fn)
    return frogs


def _SLFA(img, K):
    img = img.astype(np.int32)
    img_vec = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    frogs = Shuffled_Leaping_Frogs_Algorithm(img_vec, K)
    print('test')


def SLFA(img, K):
    T_max = 100
    J_max = 10
    rho = 0.5
    num_frogs = 50
    MAX = 255
    MIN = 0
    num_memeplexes = 10
    color_palette = np.arange(0, 256, 1)
    perm = np.arange(0, num_frogs, 1)
    # frogs = np.array([np.array([np.random.choice(color_palette, K, replace=False) for n in range(num_frogs)])
    frogs = np.random.randint(0, 255, size=(num_frogs, K, 3))

    def fit(individual):
        """
        適合度関数
        MSEまたは提案距離を用いる
        :param individual:
        :return:
        """
        quantized = CIQ_from_ColorPalettes(img, individual)
        fitness = get_mse(img, quantized)
        return 1.0 / fitness

    def calc_frogs(frog):
        tmp_fit = np.zeros((len(frog)))
        for j in range((len(frog))):
            tmp_fit[j] = fit(frog[j])

        return tmp_fit

    for t in range(T_max):
        fitness = calc_frogs(frogs)
        shuffled_index = np.random.permutation(perm)
        shuffled_frogs = frogs[shuffled_index]
        shuffled_fitness = fitness[shuffled_index]
        global_best_index = np.argmax(fitness)

        print('best frog: {} \n best score: {}'.format(shuffled_frogs[global_best_index],
                                                       shuffled_fitness[global_best_index]))

        # divide all frogs into several memeplexes
        assert num_frogs % num_memeplexes == 0, "wrong setting of num_frogs or num_memeplex"

        for bd in range(num_memeplexes, num_frogs, num_memeplexes):
            index_bius = bd - num_memeplexes
            mem_best_index = np.argmax(shuffled_fitness[index_bius: bd])
            mem_worst_index = np.argmin(shuffled_fitness[index_bius: bd])
            # compute Eqs.(1) and (2)
            D = rho * (shuffled_fitness[mem_best_index + index_bius] - shuffled_fitness[mem_worst_index + index_bius])
            next_x_worst = shuffled_frogs[mem_worst_index + index_bius] + D
            next_x_worst_fit = fit(next_x_worst)

            if next_x_worst_fit < shuffled_fitness[mem_worst_index + index_bius]:
                # apply Eqs.(1) and (3)
                D = rho * (shuffled_frogs[-1] - shuffled_frogs[mem_worst_index + bd])
                next_x_worst = shuffled_frogs[mem_worst_index + bd] + D
                next_x_worst_fit = fit(next_x_worst)

                if next_x_worst_fit < shuffled_fitness[mem_worst_index + bd]:
                    # Move the worst frog to a random position
                    # next_x_worst = np.random.choice(color_palette, K, replace=False)
                    next_x_worst = np.random.randint(0, 255, size=(K, 3))
            shuffled_frogs[mem_worst_index + bd] = next_x_worst


def SFLA(fit, create_frog, n_frogs=100, n_mem=10, T_max=100, J_max=100, rho=0.5):
    perm = np.arange(0, n_frogs, 1)
    frogs = np.array([create_frog() for _ in range(n_frogs)])

    def eval_frogs(i_frogs):
        score = np.zeros((len(i_frogs)))
        for j in range((len(i_frogs))):
            score[j] = fit(i_frogs[j])

        return score

    for t in range(T_max):
        fitness = eval_frogs(frogs)
        shuffled_index = np.random.permutation(perm)
        shuffled_frogs = frogs[shuffled_index]
        shuffled_fitness = fitness[shuffled_index]
        global_best_index = np.argmax(fitness)

        print('best frog: {} \n best score: {}'.format(shuffled_frogs[global_best_index],
                                                       shuffled_fitness[global_best_index]))

        # divide all frogs into several memeplexes
        assert n_frogs % n_mem == 0, "wrong setting of num_frogs or num_memeplex"

        for bd in range(n_mem, n_frogs, n_mem):
            index_bius = bd - n_mem
            mem_best_index = np.argmax(shuffled_fitness[index_bius: bd])
            mem_worst_index = np.argmin(shuffled_fitness[index_bius: bd])
            # compute Eqs.(1) and (2)
            D = rho * (shuffled_fitness[mem_best_index + index_bius] - shuffled_fitness[mem_worst_index + index_bius])
            next_x_worst = shuffled_frogs[mem_worst_index + index_bius] + D
            next_x_worst_fit = fit(next_x_worst)

            if next_x_worst_fit < shuffled_fitness[mem_worst_index + index_bius]:
                # apply Eqs.(1) and (3)
                D = rho * (shuffled_frogs[-1] - shuffled_frogs[mem_worst_index + bd])
                next_x_worst = shuffled_frogs[mem_worst_index + bd] + D
                next_x_worst_fit = fit(next_x_worst)

                if next_x_worst_fit < shuffled_fitness[mem_worst_index + bd]:
                    # Move the worst frog to a random position
                    next_x_worst = create_frog()
            shuffled_frogs[mem_worst_index + bd] = next_x_worst


def CIQ_from_ColorPalettes(img, palette):
    img_vec = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    err = np.zeros((len(palette), img_vec.shape[0], 3), dtype=np.float32)  # tem[K][65536][3]
    for i in range(len(palette)):
        err[i] = (img_vec - palette[i]) ** 2

    dist = np.sum(err, axis=2)
    dist = np.sqrt(dist)
    dis = dist ** 2
    cln = np.argmin(dis, axis=0)
    cln = cln.reshape((-1, 1))

    # convert back into uint8, and make original image
    center = np.uint8(palette)
    cln = np.uint8(cln)
    out_vec = center[cln.flatten()]
    output_img = np.reshape(out_vec, img.shape)
    return output_img


def get_mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def get_psnr(mse):
    return 20 * np.log10(255 / np.sqrt(mse))


def get_sug1_error(img1, img2, window_size, selector=0, a=10, b=0.1):
    height, width, channels = img1.shape
    local_mse = []
    local_position = []
    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            # 局所領域の誤差の算出
            current_mse = get_mse(img1[y: y + window_size, x: x + window_size],
                                  img2[y: y + window_size, x: x + window_size])
            # current_mse = np.sum(np.abs(img1[y: y + window_size, x: x + window_size] -
            #                             img2[y: y + window_size, x: x + window_size])) / np.power(window_size, 2.0)
            local_mse.append(current_mse)
            local_position.append([y, x])
    local_mse = np.array(local_mse)
    local_position = np.array(local_position)
    index = np.argsort(local_mse)

    # 重みの作成
    sug1_error = 0
    weighted_sorted_error = []

    n_max = len(local_mse)
    for n in range(n_max):
        if selector == 0:
            sug1_error += func1(float(n / n_max)) * local_mse[index[n_max - n]]
        elif selector == 1:
            tmp = func2(a, float(n / n_max)) * local_mse[index[n_max - n - 1]]
            sug1_error += tmp
            weighted_sorted_error.append(tmp)
        elif selector == 2:
            tmp = func3(a, float(n / n_max), b) * local_mse[index[n_max - n - 1]]
            sug1_error += tmp
            weighted_sorted_error.append(tmp)
        elif selector == 3:
            tmp = func4(a, float(n / n_max)) * local_mse[index[n_max - n - 1]]
            sug1_error += tmp
            weighted_sorted_error.append(tmp)

    sug1_error /= n_max

    return sug1_error, local_mse[index[::-1]], weighted_sorted_error


def func1(x):
    return 1 - x


def func2(a, x):
    b = (-1 * a + np.sqrt(np.power(a, 2) + 4 * a)) / 2
    c = 1 - (1 / b)
    return 1 / (a * x + b) + c


def func3(a, x, b=0.5):
    return 1.0 / (1.0 + np.exp(-(b - x) * a))


def func4(a, x):
    return np.power(np.e, -x * a)


def draw_sugDist():
    x = np.arange(0, 1, 0.05)
    ws = [1, 5, 10, 20, 50]
    ss = ['s', ',', 'o', 'v', '1']
    y = []
    fig = plt.figure()
    fig.patch.set_alpha(0.0)
    plt.rcParams['font.size'] = 15
    ax = fig.add_subplot(1, 1, 1)
    for w, s in zip(ws, ss):
        tmp = func3(w, x)
        ax.plot(x, tmp, label='a = {}'.format(w), marker=s)
    ax.grid(which='both', color='#999999', linestyle='solid')
    ax.legend()
    # ax.set_xlabel('')
    # ax.set_ylabel('weights')
    plt.savefig('weights_func3.png')


def CIQ_byGA(img, window_size, gen, K=16, pop=100, filename='', sug_flag=False, a=50):
    if not os.path.isdir(filename):
        os.mkdir(filename)
        print('make dir {}'.format(filename))

    save_dir = os.path.join(filename, 'CIQ_byGA_class_{}_window_{}_weight_{}'.format(K, window_size, a))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        print('make dir {}'.format(save_dir))

    # 問題の設定
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    def create_gene():
        gene = np.random.randint(0, 255, (K, 3))
        return creator.Individual(gene)

    # 個体集団の設定
    toolbox = base.Toolbox()
    toolbox.register("population", tools.initRepeat, list, create_gene)

    def fit(individual):
        """
        適合度関数
        MSEまたは提案距離を用いる
        :param individual:
        :return:
        """
        quantized = CIQ_from_ColorPalettes(img, individual)
        fitness = 0
        if sug_flag:
            fitness, _, __ = get_sug1_error(img, quantized, window_size, 1, a)
        else:
            fitness = get_mse(img, quantized)
        return fitness,

    def mutate(individual):
        point = random.randint(0, 4)
        sumple = create_gene()

        individual[point] = sumple[point]
        return individual,

    toolbox.register("evaluate", fit)
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", mutate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed()
    # 初期の個体群を生成
    pop = toolbox.population(n=(pop - 1))

    # 1個体のみKMeansによる量子化で得たカラーパレットを用いる
    _, centers = KMeans_Clastering(img, img, K)
    pop.append(creator.Individual(centers))

    CXPB, MUTPB, NGEN = 0.8, 0.1, gen  # 交差確率、突然変異確率、進化計算のループ回数

    print("Start of evolution")

    # 初期の個体群の評価
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    hist = []
    pop_hist = []
    # 進化計算開始
    fits_for_graph = []
    for g in range(NGEN):
        # print("-- Generation %i --" % g)

        # 次世代の個体群を選択
        offspring = toolbox.select(pop, len(pop))
        # 個体群のクローンを生成
        offspring = list(map(toolbox.clone, offspring))

        # 選択した個体群に交差と突然変異を適応する
        # 偶数番目と奇数番目の個体を取り出して交差
        # offspring.sort()
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 適合度が計算されていない個体を集めて適合度を計算
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # print("  Evaluated %i individuals" % len(invalid_ind))

        # 次世代群をoffspringにする
        pop[:] = offspring

        # すべての個体の適合度を配列にする
        fits = [ind.fitness.values[0] for ind in pop]

        best_ind = tools.selBest(pop, 1)[0]

        if g % 10 == 0 or g == (NGEN - 1):
            print("%s gen: Best individual score is %s" % (g, best_ind.fitness.values))

            filename = os.path.join(save_dir, 'class_{}_MSE_gen{}.bmp'.format(K, g))
            quantized = CIQ_from_ColorPalettes(img, best_ind)
            cv2.imwrite(filename, quantized)
            mse = get_mse(img, quantized)

        hist.append(best_ind.fitness.values[0])
        fits_for_graph.append(best_ind.fitness.values)
        pop_hist.append([best_ind])

    # print("-- End of (successful) evolution --")

    gens_index = np.arange(0, NGEN)

    fig = plt.figure()
    gen_ax = fig.add_subplot(1, 1, 1)
    gen_ax.plot(gens_index, fits)
    gen_ax.set_xlabel('Generations')
    gen_ax.set_ylabel('Fitness')
    plt.tight_layout()
    res_filename = os.path.join(save_dir + 'fitness.bmp')
    plt.savefig(res_filename)

    arr = np.array(fits)
    df = pd.DataFrame(arr)
    df.column = ['fitness']
    plt.close(fig)

    csv_name = os.path.join(save_dir, 'fitness.csv')
    df.to_csv(csv_name)
    print('save {} csv'.format(csv_name))

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual score is %s" % (best_ind.fitness.values))

    return best_ind


def evaluate_CIQ_byIterativeKMeans():
    dir = 'sumple_img'

    imgs_name = os.listdir(dir)
    for img_name in imgs_name:
        file_path = os.path.join(dir, img_name)
        img = cv2.imread(file_path).astype(np.float32)
        name = img_name.replace('.bmp', '')
        for c in [8, 16]:
            for window in [4, 8]:
                Test_Iterative_KMeans(img, window, 500, c, name)
                Test_ISRandCIQ_MedianCut(img, window, 500, c, name)


def evaluate_Resizing_onCIQ():
    # 少数色を考慮しながらの、KMeansクラスタリングによる画像の減色化に、リサイズした後に局所的な誤差を計算することは有効であるかを検証する
    img = cv2.imread('sumple_img/pencils.bmp').astype(np.float32)
    for c in [16, 32]:
        for window in [4, 8, 16]:
            start = time.time()
            Test_Iterative_KMeans_byResizing(img, window, 10, c)
            end = time.time()
            print('time: {}'.format(end - start))
            start = time.time()
            Test_Iterative_KMeans(img, window, 10, c)
            end = time.time()
            print('time: {}'.format(end - start))


def evaluate_SRandCIQ():
    dir = 'sumple_img'

    imgs_name = os.listdir(dir)
    for img_name in imgs_name:
        file_path = os.path.join(dir, img_name)
        img = cv2.imread(file_path).astype(np.float32)
        name = img_name.replace('.bmp', '')
        for sr_num in [2]:
            for r in [2]:
                for c in [8, 16, 32]:
                    for window in [4, 8, 16]:
                        Test_ISRandCIQ_KMeans(img, window, 300, name, r, c, sr_num)
                        # Test_ISRandCIQ_MedianCut(img, window, 300, name, r, c, sr_num)


def evaluate_CIQ_byGA():
    dir = 'sumple_img'
    imgs_name = os.listdir(dir)
    for img_name in imgs_name:
        file_path = os.path.join(dir, img_name)
        img = cv2.imread(file_path).astype(np.float32)
        name = img_name.replace('.bmp', '')
        for c in [16]:
            for window in [4, 8]:
                CIQ_byGA(img, window, 100, c, 100, name, True, 2)


def evaluate_CIQ_bySFLA():
    dir = 'sumple_img'
    imgs_name = os.listdir(dir)
    for img_name in imgs_name:
        file_path = os.path.join(dir, img_name)
        img = cv2.imread(file_path).astype(np.float32)
        name = img_name.replace('.bmp', '')
        for c in [16]:
            SLFA(img, c)


def MinMaxBySFLA():
    def fit(frog):
        return np.sum(frog)

    def create_frog():
        return np.random.randint(0, 2, 10)

    SFLA(fit, create_frog)


def get_allMSEandSug_inDir(org_img, save_dir, file_name, window_size):
    index = 1
    while os.path.isfile(os.path.join(save_dir, '{}'.format(index) + file_name)):
        img = cv2.imread(os.path.join(save_dir, '{}'.format(index) + file_name))
        mse = get_mse(org_img, img)
        sug1_error = get_sug1_error(org_img, img, window_size)
        print("{} \t MSE: {}\t Sug_1_error: {}".format('{}'.format(index) + file_name, mse, sug1_error))
        index += 1


def calculate_KMeansCIQImgMSEandSug(c, window):
    img = cv2.imread('sumple_img/pencils.bmp').astype(np.float32)
    get_allMSEandSug_inDir(img, 'class_{}_window_{}'.format(c, window), 'times_output.bmp', window)


def enlarge_and_save():
    imgs = os.listdir("sumple_img/")
    for index in imgs:
        img = cv2.imread("sumple_img/" + index)
        enlarged = cv2.resize(img, (int(img.shape[1] * 2), int(img.shape[0] * 2)), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("enlarged_" + index, enlarged)


def generate_average():
    dir = 'sumple_img'
    result_file_name = 'bests.csv'

    imgs_name = os.listdir(dir)
    count = 0
    sum_df = None
    for img_name in imgs_name:
        try:
            file_path = os.path.join(dir, img_name)
            name = img_name.replace('.bmp', '')
            env_names = os.listdir(name)
            print('start loading {} best scores'.format(name))
            for env in env_names:
                env_path = os.path.join(name, env)
                enlarged_names = os.listdir(env_path)
                for enlarged_name in enlarged_names:
                    result_path = os.path.join(env_path, enlarged_name, result_file_name)
                    if sum_df is None:
                        sum_df = pd.read_csv(result_path)
                        count += 1
                    else:
                        sum_df += pd.read_csv(result_path)
                        count += 1
        except:
            break
    sum_df.iloc[:, 1:] = sum_df.iloc[:, 1:].div(count)
    sum_df.iloc[1:, :].to_csv("average_result.csv")


if __name__ == '__main__':
    # generate_average()
    # draw_sugDist()
    # enlarge_and_save()
    # evaluate_CIQ_byIterativeKMeans()
    MinMaxBySFLA()
    # evaluate_CIQ_bySFLA()
    # evaluate_SRandCIQ()
    # evaluate_CIQ_byGA()
    # calculate_KMeansCIQImgMSEandSug(16, 4)
