import numpy as np
import os
import cv2
import time
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import minmax_scale
from skimage.measure import compare_nrmse, compare_psnr, compare_ssim
from img_util import get_saliency_upper_th, make_colormap, get_saliency_hist, get_numcolors, get_spectralresidual, \
    get_saliency_lower_th, mapping_pallet_to_img, compare_labmse, get_allcolors_from_img, make_colormap
from btpd import BTPD, SMBW_BTPD, BTPD_WTSE, BTPD_PaletteDeterminationFromSV, BTPD_InitializationFromSv, \
    BTPD_InitializationFromIncludingSv, BTPD_LimitationSv, BTPD_WTSE_LimitationSv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def CIEDE76(img1, img2):
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LUV)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LUV)

    # Cab_1 = np.sqrt(img1_lab[1] ** 2 + img1_lab[2] ** 2)
    # Cab_2 = np.sqrt(img2_lab[1] ** 2 + img2_lab[2] ** 2)
    #
    # hab_1 = np.arctan(img1_lab[2] / img1_lab[1])
    # hab_2 = np.arctan(img2_lab[2] / img2_lab[1])

    dist = compare_nrmse(img1_lab, img2_lab)
    return dist


def CIQ_test(ciq, test_name, test_img='sumple_img', **test_config):
    DIR = test_img
    SAVE = test_name
    imgs = os.listdir(DIR)
    trans_flag = test_config['trans_flag']
    code = test_config['trans_code']
    inverse_code = test_config['trans_inverse_code']
    view_distribution = test_config['view_distribution']
    save_tmp_imgs = test_config['save_tmp_imgs']
    view_importance = test_config['view_importance']
    ciq_error_eval = test_config['ciq_error_eval']
    importance_eval = test_config['importance_eval']
    mapping = test_config['mapping']

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        root, ext = os.path.splitext(img_path)
        save_path = os.path.join(SAVE, root)
        img = cv2.imread(path)
        groups = None
        tmp_sm = None
        palette = None
        try:
            ciq_status = {'img_path': root,
                          'num': num}
            st = time.time()
            dict = ciq(img, **ciq_status)
            en = time.time()
            palette = dict['palette']
            if view_distribution:
                groups = dict['groups']
            if save_tmp_imgs:
                tmp_imgs = dict['save_imgs']

        except KeyboardInterrupt as e:
            print(f'Error {e}, in{img_path}')

        if trans_flag:
            luv_img = cv2.cvtColor(img, code)
            mapped = mapping(luv_img, palette)
            mapped = np.reshape(mapped, newshape=img.shape)
            mapped = cv2.cvtColor(mapped, inverse_code)
        else:
            mapped = mapping(img, palette)
        mapped = np.reshape(mapped, newshape=img.shape)

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # eval
        eval_array = []
        eval_indices = []
        for eval in ciq_error_eval:
            eval_errors = eval(img, mapped)
            for eval_error in eval_errors:
                score = eval_error['error']
                eval_index = eval_error['index']
                eval_array.append(score)
                eval_indices.append(eval_index)

        # Importance
        if importance_eval:
            for statistics in ['sum_imp', 'mean_imp']:
                importance_mat = get_img_importance(root, statistics=statistics)
                eval_errors = importance_eval(img, mapped, importance_mat)
                for eval_error in eval_errors:
                    score = eval_error['error']
                    eval_index = f"{statistics}_{eval_error['index']}"
                    eval_array.append(score)
                    eval_indices.append(eval_index)

        df = pd.DataFrame([[img_path, en - st, *eval_array]], columns=['img_name', 'running time', *eval_indices])
        csv_path = os.path.join(SAVE, '{}_scores.csv'.format(test_name))

        if num != 0:
            pre_csv = pd.read_csv(csv_path, index_col=0)
            df = pre_csv.append(df)
        df.to_csv(csv_path)

        print('{}/{} {} , by {}, calc time {}s'.format(num, len(imgs), img_path, test_name, en - st))
        # mapped = mapping_pallet_to_img(img, q)
        mapped_path = os.path.join(save_path, img_path)
        cv2.imwrite(mapped_path, mapped)

        # save color map
        color_map_path = os.path.join(save_path, 'map_' + img_path)
        color_map = make_colormap(palette).astype(np.uint8)

        if trans_flag:
            color_map = cv2.cvtColor(color_map, inverse_code)

        cv2.imwrite(color_map_path, color_map)

        # save residual image between org and quantized
        residual = cv2.cvtColor(img - mapped, cv2.COLOR_BGR2GRAY)
        # residual = cv2.blur(residual, (5, 5))
        residual_path = os.path.join(save_path, 'residual_' + img_path)
        cv2.imwrite(residual_path, residual)

        # save tmp imgs
        if save_tmp_imgs:
            for tmp_img in tmp_imgs:
                img_name = tmp_img['filename']
                tmp_img_bin = tmp_img['img']
                filepath = os.path.join(save_path, img_name)
                cv2.imwrite(filepath, tmp_img_bin)
                print(f'save {filepath}')

        if view_distribution:
            # save_color_distribution(groups, save_path, 'Dist_' + root + '.jpg')
            make_color_map_with_representation(groups[0], groups[1], save_path)

        if view_importance:
            pass


def save_mse_in_eachSaliency(img, mapped, save_path=None, filename=None):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist, bins, sm = get_saliency_hist(lab_img, sm='SR')
    rgb_mse_sv = np.zeros(shape=256)
    for num in range(0, 256):
        position = np.where(sm == num)
        if len(position) == 0:
            continue
        # rgb_mse = (np.square(img[position] - mapped[position])).mean()
        rgb_mse = compare_nrmse(img[position], mapped[position])
        rgb_mse_sv[num] = rgb_mse

    if save_path or filename:
        mse_saliency_path = os.path.join(save_path, filename)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # 何もプロットしていないAxesでもAxisは自動的に作られる
        ax.scatter(bins, rgb_mse_sv, c='red', label='RGB_MSE', marker='.')
        ax.set_title("MSE in each saliency")
        ax.set_xlabel("saliency")
        ax.set_ylabel("mean of MSE")
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid()
        plt.savefig(mse_saliency_path)
        plt.close()

    return rgb_mse_sv


def make_color_distribution_img(pixels, img_name, statistics=()):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')  # 何もプロットしていないAxesでもAxisは自動的に作られる
    colors = pixels / 255.0
    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=colors, marker=',', label='pixels')
    # ax.set_title(img_name)
    ax.set_xlabel("B")
    ax.set_ylabel("G")
    ax.set_zlabel("R")
    ax.set_xlim([0, 255])
    ax.set_ylim([0, 255])
    ax.set_zlim([0, 255])

    ax.grid()

    cell = []
    indices = []
    for stat in statistics:
        label = stat['label']
        func = stat['func']
        cell.append(func(pixels))
        indices.append(label)
    if statistics:
        ax.table(cellText=[cell], colLabels=[indices], loc='bottom')

    # fig.patch.set_alpha(0.0)
    return fig


def save_color_distribution(groups, save_path, filename):
    width = 1
    while width ** 2 < len(groups):
        width += 1
    for num, group in enumerate(groups):
        img_name = f'{len(group)}colors_distribution_n{num + 1}'
        fig = make_color_distribution_img(group, img_name)
        filepath = os.path.join(save_path, f'{img_name}.jpg')
        fig.savefig(filepath, bbox_inches="tight", transparent=True)
        print(f'save {filepath}')
        plt.close()


def make_color_map_with_representation(groups, representations, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')  # 何もプロットしていないAxesでもAxisは自動的に作られる
    for pixels, representaion in zip(groups, representations):
        color = np.array(representaion / 255.0)
        ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=color, marker=',', label='pixels')
        # ax.set_title(img_name)
        ax.set_xlabel("B")
        ax.set_ylabel("G")
        ax.set_zlabel("R")
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])

    for angle in range(0, 360, 45):
        for z_angle in range(0, 180, 45):
            ax.view_init(z_angle, angle)
            plt.draw()
            plt.pause(.001)

            img_name = f'{len(groups)}representaions_{angle}_{z_angle}'
            filepath = os.path.join(save_path, f'{img_name}.jpg')
            fig.savefig(filepath, bbox_inches="tight", transparent=True)
    print(f'save {filepath}')
    # plt.show()
    plt.close()


def save_color_importanceerror(img, mapped, importance, save_path, filename):
    pass


def get_img_importance(img_path, statistics='sum_imp'):
    SAVE = 'Importance_Map'
    img_name, ext = os.path.splitext(img_path)
    # save_path = os.path.join(SAVE, img_name, f'{img_name}.csv')
    save_path = os.path.join(SAVE, img_name, f'premapped_{statistics}_importance.csv.')
    csv = pd.read_csv(save_path, index_col=0, dtype=np.float32)
    csv = csv / np.max(csv)
    return csv


def CIQ_test_gradually():
    DIR = 'sumple_img'
    SAVE = 'BTPD_256_16'
    M = [256, 16]

    def ciq(img):
        S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, img.shape[2]))
        for m in M:
            S = np.reshape(S, newshape=(len(S), 1, 3)).astype(np.uint64)
            print('len S: {}'.format(len(S)))
            q, root = BTPD(S, m)
            S = q
        return q

    CIQ_test(ciq, SAVE, DIR)


def CIQ_test_sup1(M=[16, 32], DIR=['sumple_img'], R=[0.1]):
    for dir in DIR:
        for m in M:
            for r in R:
                code = cv2.COLOR_BGR2LAB
                inverse_code = cv2.COLOR_LAB2BGR

                def ciq(img):
                    trans_img = cv2.cvtColor(img, code)
                    extract, parted_extract, zeros, min_sm = get_saliency_upper_th(trans_img, r, sm='SR')
                    print('min_sm: {}'.format(min_sm))
                    org_S = np.reshape(extract, newshape=(len(extract), 1, 3)).astype(np.uint64)
                    # cv2.imshow('test', zeros)
                    # cv2.waitKey(0)
                    q, root = BTPD(extract, m)
                    leaves = root.get_leaves()
                    groups = []
                    for leaf in leaves:
                        index = leaf.get_data()['index']
                        pixels = org_S[index]
                        pixels = np.reshape(pixels, newshape=(len(pixels), 3))
                        groups.append(pixels)
                    return q, np.array(groups)
                SAVE = 'sup1_M{}_R{}_{}'.format(m, r, dir)
                CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code,
                     view_distribution=True)


def CIQ_test_sup2():
    # 顕著度を何分割化し(高顕著度であれば小さく分割し，低顕著度であれば大きく分割する)，
    # 各分割に属する色で，何色かに量子化する
    M = 16
    DIR = 'sumple_img'
    SAVE = 'sup2'
    PARTITION = 2

    def ciq(img):
        num_colors = get_numcolors(img)
        R = num_colors * 0.0006
        hist, bins, sm = get_saliency_hist(img, sm='SR')
        # 顕著度の中央値で2つに分割する
        med = np.median(sm)
        upper_indices = np.where(sm >= med)
        lower_indices = np.where(sm < med)

        upper_pix = img[upper_indices]
        lower_pix = img[lower_indices]

        var1 = np.mean(np.var(upper_pix, axis=0))
        var2 = np.mean(np.var(lower_pix, axis=0))
        tmp1 = var1 ** 2
        tmp2 = var2 ** 2

        sumple_ratio = int(M * R)
        ratio1 = int(tmp1 * sumple_ratio / (tmp1 + tmp2))
        ratio2 = int(tmp2 * sumple_ratio / (tmp1 + tmp2))
        # ratio1 = 512
        # ratio2 = 256
        print('num colors: {} \t sumple ratio: {}'.format(num_colors, sumple_ratio))
        print('q1 var{} \t {} colors -- > {} colors'.format(var1, get_numcolors(upper_pix), ratio1))
        q1 = BTPD(upper_pix, ratio1)

        print('q2 var{} \t {} colors -- > {} colors'.format(var2, get_numcolors(lower_pix), ratio2))
        q2 = BTPD(lower_pix, ratio2)

        q = np.append(q1, q2, axis=0)

        q = BTPD(q, M)
        return q

    CIQ_test(ciq, SAVE, DIR)


def CIQ_test_sup3():
    # 少数色でも顕著度が高ければ保存する
    # --> R = 0 - 0.1 の画素数をR = 0.4 - 0.5の画素数ほどに増加させる
    M = 16
    R = 16
    DIR = 'sumple_img'
    SAVE = 'Quantized_based_on_SM_uniformly_BTPD_R{}'.format(R)

    def ciq(img):
        __, parted_extract = get_saliency_upper_th(img, 1.0)

        q = []
        for num, extract in enumerate(parted_extract):
            n_colors = get_numcolors(extract)
            if n_colors > (R * 2):
                print('{}th prequantized: {} --> {}'.format(num, n_colors, R))
                S = np.reshape(extract, newshape=(len(extract), img.shape[2]))
                palette = BTPD(S, R)
                q.extend(palette)
            elif n_colors > 0:
                extract = np.reshape(extract, newshape=(len(extract), 1, 3))
                q.extend(extract)
        q = np.array(q)
        q = BTPD(q, M)
        return q

    CIQ_test(ciq, SAVE, DIR)


def CIQ_test_sup4():
    # 一様乱数を生成し，顕著度に比例して選ばれやすい
    # のようなルールに従って，画素をサンプリングする
    M = 16
    R = 16
    N_SUMPLE = 1024
    DIR = 'sumple_img'
    SAVE = 'Quantized_based_on_SM_uniformly_BTPD_R{}'.format(R)

    def sumpling(S, Sv, num):
        sumple = []
        sumpled_num = np.random.randint(0, 256, num)

    def ciq(img):
        __, parted_extract = get_saliency_upper_th(img, 1.0)

        q = []
        for num, extract in enumerate(parted_extract):
            n_colors = get_numcolors(extract)
            if n_colors > (R * 2):
                print('{}th prequantized: {} --> {}'.format(num, n_colors, R))
                S = np.reshape(extract, newshape=(len(extract), img.shape[2]))
                palette = BTPD(S, R)
                q.extend(palette)
            elif n_colors > 0:
                extract = np.reshape(extract, newshape=(len(extract), 1, 3))
                q.extend(extract)
        q = np.array(q)
        q = BTPD(q, M)
        return q

    CIQ_test(ciq, SAVE, DIR)


def ciq_eval_set():
    # NRMSE, PSNR, Lab_NRMSE, SSIM
    def trad_dist(org, mapped):
        nrmse = compare_nrmse(org, mapped)
        psnr = compare_psnr(org, mapped)
        lab_nrmse = compare_nrmse(org, mapped)
        ssim = compare_ssim(org, mapped, multichannel=True)

        eval_list = [{'error': nrmse, 'index': 'NRMSE'},
                    {'error': psnr, 'index': 'PSNR'},
                    {'error': lab_nrmse, 'index': 'LAB_NRMSE'},
                    {'error': ssim, 'index': 'SSIM'}]

        return eval_list

    def get_rgb_topdist(org, mapped):
        rgb_mse = np.linalg.norm(org - mapped, axis=2).flatten() / org.shape[2]
        rgb_mse_top_dist = np.sort(rgb_mse)[::-1]
        eval_list = []
        for part in [.01, .05, .1]:
            index = int(part * len(rgb_mse_top_dist))
            mean = np.mean(rgb_mse_top_dist[:index])
            eval = {'error': mean, 'index': f'Top Dist {int(part * 100)}% RGB Error '}
            eval_list.append(eval)
        return eval_list

    def get_rgb_sv_topdist(org, mapped):
        sv_dist = save_mse_in_eachSaliency(org, mapped)
        nan_index = np.where(np.isnan(sv_dist))
        # len_without_nan = len(sv_dist) - len(nan_index)
        sv_dist[nan_index] = .0
        sv_dist = sv_dist[::-1]
        eval_list = []
        for part in [.01, .05, .1]:
            index = int(part * len(sv_dist))
            mean = np.mean(sv_dist[:index])
            eval = {'error': mean, 'index': f'Top Dist {int(part * 100)}% Sv Error '}
            eval_list.append(eval)
        return eval_list

    return [trad_dist, get_rgb_topdist, get_rgb_sv_topdist]


def get_importance(org, mapped, importance_mat):
    # width = org.shape[1]
    eval_list = []
    if type(importance_mat) == pd.DataFrame:
        importance_mat = importance_mat.values
    all_colors = get_allcolors_from_img(mapped)
    colors_importance = [importance_mat[np.where(mapped == color)[:2]][0] for color in all_colors]
    sorted_index = np.argsort(colors_importance)
    sorted_colors = all_colors[sorted_index]
    # sorted_position = [np.array([index // width, index % width]) for index in sorted_index[::-1]]
    importance_top_ratio = [.01, .05, .1]
    dist_img = np.linalg.norm(org - mapped, axis=2) / org.shape[2]
    each_color_dist = [dist_img[np.where(mapped == color)[:2]] for color in sorted_colors[::-1]]
    each_color_dist = [np.sum(dists) / len(dists) for dists in each_color_dist]
    sum_imp = .0
    imp_n = 0
    for n, dist in enumerate(each_color_dist):
        sum_imp += dist
        if (n + 1) > (importance_top_ratio[imp_n] * len(sorted_index)):
                eval = {'error': sum_imp / (n + 1), 'index': f'Top Importance {importance_top_ratio[imp_n] * 100}%'}
                eval_list.append(eval)
                imp_n += 1
        if imp_n >= len(importance_top_ratio):
            break
    return eval_list


def get_importance_error(org, mapped, importance_mat):
    # width = org.shape[1]
    eval_list = []
    if type(importance_mat) == pd.DataFrame:
        importance_mat = importance_mat.values

    flattened_mat = importance_mat.flatten()
    flattened_org = np.reshape(org, (flattened_mat.size, 3))
    flattened_mapped = np.reshape(mapped, flattened_org.shape)
    sorted_indices = np.argsort(flattened_mat)
    sorted_indices = sorted_indices[::-1]
    sorted_mat = flattened_mat[sorted_indices]
    sorted_org = flattened_org[sorted_indices]
    sorted_mapped = flattened_mapped[sorted_indices]

    importance_top_ratio = [.01, .05, .1, .3]

    for ratio in importance_top_ratio:
        th = round(len(flattened_mat) * ratio)
        nrmse = compare_nrmse(sorted_org[:th], sorted_mapped[:th])
        eval = {'error': nrmse, 'index': f'Top Importance {ratio * 100}%'}
        eval_list.append(eval)
    return eval_list


def get_importance_error_individually_color(org, mapped, importance_mat):
    # width = org.shape[1]
    eval_list = []
    if type(importance_mat) == pd.DataFrame:
        importance_mat = importance_mat.values

    uniq_imp = np.unique(importance_mat)
    sorted_imp = np.sort(uniq_imp)[::-1]
    indices = [np.where(importance_mat == imp) for imp in sorted_imp]
    sorted_mapped = np.array([mapped[ind] for ind in indices])
    sorted_org = np.array([org[ind] for ind in indices])
    n_colors = np.array([len(pixels) for pixels in sorted_org])
    sum_p = np.sum(n_colors)

    importance_top_ratio = [.01, .05, .1, .3]

    for ratio in importance_top_ratio:
        th = round(len(uniq_imp) * ratio)
        # print(ratio)
        current_org = np.concatenate(sorted_org[:th])
        current_mapped = np.concatenate(sorted_mapped[:th])
        nrmse = compare_nrmse(current_org, current_mapped)
        eval = {'error': nrmse, 'index': f'Top Importance {ratio * 100}%'}
        eval_list.append(eval)
    return eval_list


def CIQ_test_sup5(M=[16, 32], DIR=['sumple_img'], R={0.2, 0.25}):
    # SMを二値化して量子化する
    for dir in DIR:
        for m in M:
            for r in R:
                def ciq(img):
                    upper_extract, _, __ = get_saliency_upper_th(img, r, sm='SR')
                    lower_extract, _, __ = get_saliency_lower_th(img, r, sm='SR')
                    q1 = BTPD(upper_extract, m)
                    q2 = BTPD(lower_extract, m)
                    q = np.append(q1, q2, axis=0)
                    q = BTPD(q, m)
                    return q
                SAVE = 'sup5_M{}_R{}_{}'.format(m, r, dir)
                CIQ_test(ciq, SAVE, test_img=dir)


def CIQ_test_medianbased(M=[16, 32], DIR=['sumple_img'], R=[0,1]):
    MODE = 'UPPER'
    for dir in DIR:
        for m in M:
            def ciq(img):
                hist, bins, sm = get_saliency_hist(img, sm='SR')
                # 顕著度の中央値で2つに分割する
                med = np.median(sm)
                print('Median: {}'.format(med))
                if MODE == 'UPPER':
                    indices = np.where(sm >= med)
                else:
                    indices = np.where(sm < med)
                pix = img[indices]
                q = BTPD(pix, m)
                return q
            SAVE = 'medianbased_M{}_{}_{}'.format(m, dir, MODE)
            CIQ_test(ciq, SAVE, test_img=dir)


def CIQ_test_SMBW(M=[16], DIR=['sumple_img'], M0=[0.8]):
    for dir in DIR:
        for m in M:
            for m0 in M0:
                def ciq(img):
                    S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    _, __, Sv = get_saliency_hist(img, sm='SR')
                    Sv = np.reshape(Sv, newshape=(len(S), 1))
                    q = SMBW_BTPD(S, Sv, m, M0=m0)
                    return q

                SAVE = 'SMBW_bySR_M{}_{}_M0{}'.format(m, dir, m0)
                CIQ_test(ciq, SAVE, test_img=dir)


def CIQ_test_BTPD_withSv(M=[16], DIR=['sumple_img']):
    test_config = {
        'trans_flag': True,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': False,
        'save_tmp_imgs': True,
        'view_importance': True,
        'importance_eval': get_importance_error_individually_color,
        'ciq_error_eval': ciq_eval_set(),
        'mapping': mapping_pallet_to_img
    }
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2LAB

            def ciq(img, **ciq_status):
                trans_img = cv2.cvtColor(img, code)
                S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint32)
                org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint32)
                _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float64) / np.max(Sv_map) + 1e-10
                q, root, groups = BTPD_WTSE(S, m, Sv)

                # groupsから分割色の保存
                save_imgs = []
                all_nodes = root.get_all_nodes()
                for n, node in enumerate(all_nodes):
                    data = node.get_data()
                    np_group = np.reshape(data['S'], newshape=(data['S'].shape[0], 1, 3)).astype(np.uint8)
                    retrans_s = cv2.cvtColor(np_group, cv2.COLOR_LAB2BGR)
                    c_map = make_colormap(retrans_s, color_width=16)
                    c_map_dict = {'img': c_map, 'filename': f'{n}_colormap.jpg'}
                    save_imgs.append(c_map_dict)

                reshape_q = np.reshape(q, newshape=(m, 1, 3)).astype(np.uint8)
                retrans_q = cv2.cvtColor(reshape_q, cv2.COLOR_LAB2BGR)

                # make colormap
                colormap = make_colormap(retrans_q)

                save_imgs.extend(
                    [{'img': Sv_map, 'filename': 'tmp_Sv.jpg'},
                     {'img': colormap, 'filename': 'colormap.jpg'}]
                )

                dict = {'palette': q,
                        'save_imgs': save_imgs}
                return dict
            SAVE = 'BTPD_WTSE_M{}_{}_LAB'.format(m, dir)
            CIQ_test(ciq, SAVE, test_img=dir, **test_config)


def CIQ_test_BTPD_includingSv(M=[16], DIR=['sumple_img']):
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2LAB
            inverse_code = cv2.COLOR_LAB2BGR

            def ciq(img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                _, __, Sv = get_saliency_hist(img, sm='SR')
                Sv = np.reshape(Sv, newshape=(len(S), 1, 1)).astype(np.uint64)
                S_Sv = np.concatenate([S, Sv], axis=2)
                # Sv = 1.0 / (Sv + 1.0)
                # Sv = (255.0 - Sv) / 255.0
                # Sv = Sv / 255.0
                q = BTPD(S_Sv, m)
                return q[:, :, :3]
            SAVE = 'including_Sv_{}_{}'.format(m, dir)
            CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code)


def CIQ_test_BTPD_PaletteDeterminationFromSv(M=[16], DIR=['sumple_img']):
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2LAB
            inverse_code = cv2.COLOR_LAB2BGR

            def ciq(img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                _, __, Sv = get_saliency_hist(img, sm='SR')
                Sv = np.reshape(Sv, newshape=(len(S), 1, 1)).astype(np.uint64)
                # Sv = 1.0 / (Sv + 1.0)
                # Sv = Sv / 255.0
                q = BTPD_PaletteDeterminationFromSV(S, m, Sv)
                return q

            SAVE = 'PaletteDeterminationFromSv_m{}_{}'.format(m, dir)
            CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code)


def CIQ_test_BTPD_InitializationSv(M=[16], DIR=['sumple_img']):
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2LAB
            inverse_code = cv2.COLOR_LAB2BGR

            def ciq(img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                _, __, Sv = get_saliency_hist(img, sm='SR')
                Sv = np.reshape(Sv, newshape=(len(S), 1, 1)).astype(np.float32)
                # W = (1.0 / (Sv + 1.0)).astype(np.float32)
                Sv = Sv / 255.0
                q, root = BTPD_InitializationFromSv(S, m, Sv, W)
                leaves = root.get_leaves()
                groups = []
                for leaf in leaves:
                    index = leaf.get_data()['index']
                    pixels = org_S[index]
                    pixels = np.reshape(pixels, newshape=(len(pixels), 3))
                    groups.append(pixels)
                dict = {'palette': q,
                        'groups': groups,
                        'tmp_sm': Sv_map}
                return dict
            SAVE = 'InitSv_m{}_{}'.format(m, dir)
            CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code)


def CIQ_test_BTPD_InitializationFromIncludingSv(M=[16], DIR=['sumple_img']):
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2LAB
            inverse_code = cv2.COLOR_LAB2BGR

            def ciq(img):
                trans_img = cv2.cvtColor(img, code)
                org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                _, __, Sv = get_saliency_hist(trans_img, sm='SR')
                Sv = np.reshape(Sv, newshape=(len(S), 1, 1)).astype(np.float32)
                # W = (1.0 / (Sv + 1.0)).astype(np.float32)
                W = Sv / 255.0
                q, root = BTPD_InitializationFromIncludingSv(S, m, Sv, W)
                leaves = root.get_leaves()
                groups = []
                for leaf in leaves:
                    index = leaf.get_data()['index']
                    pixels = org_S[index]
                    pixels = np.reshape(pixels, newshape=(len(pixels), 3))
                    groups.append(pixels)
                dict = {'palette': q,
                        'groups': groups}
                return dict

            SAVE = 'InitIncludingSv_m{}_{}'.format(m, dir)
            CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code,
                     view_distribution=True)


def CIQ_test_BTPD_PreQuantize(M=[16], DIR=['sumple_img'], LIMIT=[1000], weighting=False):
    """
    二分木分割アルゴリズムにおいて，各葉の第一固有値(射影後の分散と一致するはず)がLIMITを下回るまで，
    分割を繰り返す．
    -- > 第一固有値が高ければ，その葉において多くの種類の色が含まれていることが考えられる．
        従って，その葉で生じる量子化誤差が大きくなるため，その葉に対しては分割処理を繰り返す．
    こうして量子化された画像は，LIMITを下回るくらいには色が保存されていることが考えられる．
    -- > 少数色もある程度保存されているはず
    この状態で，顕著度による重み付き二分木分割アルゴリズムを適用する．
    :param M:
    :param DIR:
    :param LIMIT:
    :return:
    """
    for dir in DIR:
        for m in M:
            for lim in LIMIT:
                code = cv2.COLOR_BGR2LAB
                inverse_code = cv2.COLOR_LAB2BGR

                def ciq(img):
                    trans_img = cv2.cvtColor(img, code)
                    org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                    Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32)

                    # pre quantize
                    if weighting:
                        q, root = BTPD_WTSE_LimitationSv(S, Sv / 255.0, lim)
                    else:
                        q, root = BTPD_LimitationSv(S, lim)
                    mapped = mapping_pallet_to_img(trans_img, q)
                    print('pre quantize {} colors'.format(len(root.get_leaves())))

                    _, __, Sv_map = get_saliency_hist(mapped, sm='SR')
                    Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32)
                    S = np.reshape(mapped, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    # W = (1.0 / (Sv + 1.0)).astype(np.float32)
                    W = Sv / 255.0
                    q, root = BTPD_WTSE(S, m, W)
                    leaves = root.get_leaves()
                    groups = []
                    for leaf in leaves:
                        index = leaf.get_data()['index']
                        pixels = org_S[index]
                        pixels = np.reshape(pixels, newshape=(len(pixels), 3))
                        groups.append(pixels)
                    dict = {'palette': q,
                            'groups': groups,
                            'tmp_sm': Sv_map}
                    return dict

                SAVE = 'PreQuantizeW_m{}_{}_L{}_LAB'.format(m, dir, lim)
                CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code,
                         view_distribution=True, save_tmpSM=True)


def CIQ_test_BTPD_PreQuantizeandSVcount(M=[16], DIR=['sumple_img'], PRE_Q=[256], DIV=[512]):
    """
    各色の重要度を算出する際の計算時間の短縮のため，一旦PRE_Q色に量子化する(事前量子化)
    その後，各色の重要度の大きさに応じて，S内のその色の数を増やす
    -- > 重要とされる色の数が相対的に増加するため保存されやすくなる
    argumentation したSで量子化する
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    for dir in DIR:
        for m in M:
            for pre_q in PRE_Q:
                for div in DIV:
                    code = cv2.COLOR_BGR2LAB
                    inverse_code = cv2.COLOR_LAB2BGR

                    def ciq(img):
                        trans_img = cv2.cvtColor(img, code)
                        org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)

                        # pre quantize
                        q, root = BTPD(S, pre_q)
                        mapped = mapping_pallet_to_img(trans_img, q)
                        # SM count in each colors
                        _, __, Sv_map = get_saliency_hist(mapped, sm='SR')
                        Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32)
                        S = np.reshape(mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                        uniq_S = np.unique(S, axis=0)
                        uniq_Sv = np.round([np.sum(Sv[np.where(color == S)[0]] / div) for color in uniq_S])
                        tile_Sv = []
                        for color, sv in zip(uniq_S, uniq_Sv):
                            tile_Sv.extend(np.tile(color, (sv, 1)))
                        tile_Sv = np.array(tile_Sv)
                        tile_S = np.reshape(tile_Sv, newshape=(len(tile_Sv), 1, 3))
                        print('pre quantize {} colors'.format(len(root.get_leaves())))

                        # W = (1.0 / (Sv + 1.0)).astype(np.float32)
                        W = Sv
                        q, root = BTPD(tile_S, m)
                        leaves = root.get_leaves()
                        groups = []
                        for leaf in leaves:
                            index = leaf.get_data()['index']
                            pixels = tile_S[index]
                            pixels = np.reshape(pixels, newshape=(len(pixels), 3))
                            groups.append(pixels)
                        # importances = uniq_S[np.argsort(uniq_Sv)]
                        dict = {'palette': q,
                                'groups': groups,
                                'tmp_sm': Sv_map}
                        return dict

                    SAVE = 'PreQuantizeSVcount_m{}_{}_q{}_div{}_2'.format(m, dir, pre_q, div)
                    CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code,
                             view_distribution=True, save_tmpSM=True, view_importance=False)


def CIQ_test_BTPD_MyPreQuantizeandSVcount(M=[16], DIR=['sumple_img'], LIMIT=[3000], DIV=[512]):
    """
    上の，LIMITを設けた事前量子化を行い，
    重要度を算出する．
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    for dir in DIR:
        for m in M:
            for lim in LIMIT:
                for div in DIV:
                    code = cv2.COLOR_BGR2LAB
                    inverse_code = cv2.COLOR_LAB2BGR

                    def ciq(img):
                        trans_img = cv2.cvtColor(img, code)
                        org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                        Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32)

                        # pre quantize
                        q, root = BTPD_WTSE_LimitationSv(S, Sv / 256.0, lim)
                        mapped = mapping_pallet_to_img(trans_img, q)
                        # SM count in each colors
                        _, __, Sv_map = get_saliency_hist(mapped, sm='SR')
                        Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32)
                        S = np.reshape(mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                        uniq_S = np.unique(S, axis=0)
                        uniq_Sv = np.round([np.sum(Sv[np.where(color == S)[0]] / div) for color in uniq_S]).astype(np.int)
                        tile_Sv = []
                        for color, sv in zip(uniq_S, uniq_Sv):
                            tile_Sv.extend(np.tile(color, (sv, 1)))
                        tile_Sv = np.array(tile_Sv)
                        tile_S = np.reshape(tile_Sv, newshape=(len(tile_Sv), 1, 3))
                        print('pre quantize {} colors'.format(len(root.get_leaves())))

                        # W = (1.0 / (Sv + 1.0)).astype(np.float32)
                        W = Sv
                        q, root = BTPD(uniq_S, m)
                        leaves = root.get_leaves()
                        groups = []
                        for leaf in leaves:
                            index = leaf.get_data()['index']
                            pixels = tile_S[index]
                            pixels = np.reshape(pixels, newshape=(len(pixels), 3))
                            groups.append(pixels)
                        # importances = uniq_S[np.argsort(uniq_Sv)]
                        dict = {'palette': q,
                                'groups': groups,
                                'tmp_sm': Sv_map}
                        return dict

                    SAVE = 'MyPreQuantizeSVcountW_m{}_{}_lim{}_div{}'.format(m, dir, lim, div)
                    CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code,
                             view_distribution=True, save_tmpSM=True, view_importance=False)


def CIQ_test_ProposalTile(M=[16], DIR=['sumple_img'], LIMIT=[3000], DIV=[1]):
    """
    重要度を，合計顕著度ではなく，その色の中で最大の顕著度とする
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    test_config = {
        'trans_flag': True,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': True,
        'save_tmp_imgs': True,
        'view_importance': True,
        'importance_eval': get_importance_error_individually_color,
        'ciq_error_eval': ciq_eval_set(),
        'mapping': mapping_pallet_to_img
    }
    for dir in DIR:
        for m in M:
            for lim in LIMIT:
                for div in DIV:
                    def ciq(img, **ciq_status):
                        trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        # trans_img = img.copy()
                        org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                        org_Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map)

                        # pre quantize
                        pre_q, root, pre_groups = BTPD_WTSE_LimitationSv(S, org_Sv, lim)
                        pre_mapped = mapping_pallet_to_img(trans_img, pre_q)
                        print(f'{len(np.unique(org_S, axis=0))}')
                        # SM count in each colors
                        _, __, Sv_map = get_saliency_hist(pre_mapped, sm='SR')
                        Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map)
                        S = np.reshape(pre_mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                        uniq_S = np.unique(S, axis=0)
                        uniq_Sv = np.round([np.sum(Sv[np.where(np.all(color == S, axis=1))[0]]) / div for color in uniq_S]).astype(np.int)

                        # only in case of sum
                        uniq_Sv = (uniq_Sv / np.min(uniq_Sv)).astype(np.int)

                        tile_Sv = []
                        for color, sv in zip(uniq_S, uniq_Sv):
                            tile_Sv.extend(np.tile(color, (sv, 1)))
                        tile_Sv = np.array(tile_Sv)
                        tile_S = np.reshape(tile_Sv, newshape=(len(tile_Sv), 1, 3))
                        print('pre quantize {} colors'.format(len(root.get_leaves())))

                        q, root, groups = BTPD(tile_S, m, visualization=False)
                        pre_mapped = cv2.cvtColor(pre_mapped, cv2.COLOR_LAB2BGR)
                        reshape_q = np.reshape(q, newshape=(m, 1, 3)).astype(np.uint8)
                        retrans_q = cv2.cvtColor(reshape_q, cv2.COLOR_LAB2BGR)
                        reshape_pre_q = np.reshape(pre_q, newshape=(len(pre_q), 1, 3)).astype(np.uint8)
                        retrans_pre_q = cv2.cvtColor(reshape_pre_q, cv2.COLOR_LAB2BGR)
                        dict = {'palette': q,
                                'groups': [retrans_pre_q[:, 0, :], retrans_q[:, 0, :]],
                                'save_imgs': [{'img': Sv_map, 'filename': 'tmp_Sv.jpg'},
                                              {'img': pre_mapped, 'filename': 'pre_mapped.jpg'}]}
                        return dict

                    SAVE = 'ProposalTile_m{}_{}_lim{}_div{}_LAB'.format(m, dir, lim, div)
                    CIQ_test(ciq, SAVE, test_img=dir, **test_config)


def CIQ_test_ProposalSvSumWeight(M=[16], DIR=['sumple_img'], LIMIT=[3000]):
    """
    重要度を，合計顕著度ではなく，その色の中で最大の顕著度とする
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    test_config = {
        'trans_flag': True,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': False,
        'save_tmp_imgs': False,
        'view_importance': False,
        'importance_eval': False,
        'ciq_error_eval': ciq_eval_set(),
        'mapping': mapping_pallet_to_img
    }
    for dir in DIR:
        for m in M:
            for lim in LIMIT:
                test_title = 'ProposalSvSumWeight_m{}_{}_lim{}_LAB'.format(m, dir, lim)

                def ciq(img, **ciq_status):
                    trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    # trans_img = img.copy()
                    org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                    org_Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map) + 1e-10
                    # pre quantize
                    # pre_q, root, pre_groups = BTPD_WTSE_LimitationSv(S, org_Sv, lim)
                    pre_q, root, pre_groups = BTPD_WTSE(S, m * 16, org_Sv)
                    pre_mapped = mapping_pallet_to_img(trans_img, pre_q)
                    print(f'{len(np.unique(org_S, axis=0))}')
                    # SM count in each colors
                    _, __, Sv_map = get_saliency_hist(pre_mapped, sm='SR')
                    Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map) + 1e-10
                    S = np.reshape(pre_mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                    # scale invarianceに
                    Sv = Sv / (img.shape[0] * img.shape[1])

                    # 重要度において，顕著度Aの画素1つ == 顕著度0.5Aの画素2つとなってるので，画像サイズに応じて可変にする
                    # 顕著度Aの画素1つ ==
                    # Sv = np.power(Sv, 2.3)
                    uniq_S = np.unique(S, axis=0)
                    uniq_Sv_array = [Sv[np.all(color == S, axis=2)[:, 0]] for color in uniq_S]
                    # sum1 = np.sum([len(sv) for sv in uniq_Sv_array])
                    # len_max = np.max([len(sv_array) for sv_array in uniq_Sv])
                    # uniq_Sv = np.array([np.mean(sv_array) * (np.sum(sv_array) / len_max) for sv_array in uniq_Sv])
                    n_mean = np.mean([len(sv) for sv in uniq_Sv_array])
                    # weight = np.array([(len(sv) / n_mean) if (len(sv) > n_mean) else 1.0 for sv in uniq_Sv_array])\
                    #     .astype(np.float32)
                    # weight = np.reshape(weight, newshape=uniq_Sv.shape)
                    uniq_Sv = np.array([sv[0] for sv in uniq_Sv_array])
                    # imp = uniq_Sv * weight
                    # uniq_Sv = np.array([sv_array[0] * (len(sv_array) / n_mean) for sv_array in uniq_Sv])
                    imp = np.array([np.sum(sv_array) for sv_array in uniq_Sv])
                    imp = np.reshape(imp, newshape=(imp.shape[0], 1, 1)).astype(np.float32)

                    # only in case of sum
                    print('pre quantize {} colors'.format(len(root.get_leaves())))
                    q, root, groups = BTPD_WTSE(uniq_S, m, imp)

                    # groupsから分割色の保存
                    save_imgs = []
                    # all_nodes = root.get_all_nodes()
                    # for n, node in enumerate(all_nodes):
                    #     data = node.get_data()
                    #     np_group = np.reshape(data['S'], newshape=(data['S'].shape[0], 1, 3)).astype(np.uint8)
                    #     retrans_s = cv2.cvtColor(np_group, cv2.COLOR_LAB2BGR)
                    #     c_map = make_colormap(retrans_s)
                    #     c_map_dict = {'img': c_map, 'filename': f'{n}_colormap.jpg'}
                    #     save_imgs.append(c_map_dict)

                    # make colormap
                    indices = np.argsort(uniq_Sv)[::-1]
                    tmp_sv = uniq_Sv[indices]
                    # reshape_S = np.reshape(uniq_S, newshape=(len(indices), 1, 3))
                    retrans_S = cv2.cvtColor(uniq_S.astype(np.uint8), cv2.COLOR_LAB2BGR)
                    colormap = make_colormap(retrans_S[indices])

                    pre_mapped = cv2.cvtColor(pre_mapped, cv2.COLOR_LAB2BGR)
                    reshape_q = np.reshape(q, newshape=(m, 1, 3)).astype(np.uint8)
                    retrans_q = cv2.cvtColor(reshape_q, cv2.COLOR_LAB2BGR)
                    reshape_pre_q = np.reshape(pre_q, newshape=(len(pre_q), 1, 3)).astype(np.uint8)
                    retrans_pre_q = cv2.cvtColor(reshape_pre_q, cv2.COLOR_LAB2BGR)

                    save_imgs.extend(
                        [{'img': Sv_map, 'filename': 'tmp_Sv.jpg'},
                         {'img': pre_mapped, 'filename': 'pre_mapped.jpg'},
                         {'img': colormap, 'filename': 'colormap.jpg'}]
                    )

                    dict = {'palette': q,
                            'groups': [retrans_pre_q[:, 0, :], retrans_q[:, 0, :]],
                            'save_imgs':save_imgs}
                    return dict
                CIQ_test(ciq, test_title, test_img=dir, **test_config)


def CIQ_test_PreQuantization(M=[16], DIR=['sumple_img']):
    """
    重要度を，合計顕著度ではなく，その色の中で最大の顕著度とする
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    test_config = {
        'trans_flag': True,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': False,
        'save_tmp_imgs': True,
        'view_importance': False,
        'importance_eval': False,
        'ciq_error_eval': ciq_eval_set(),
        'mapping': mapping_pallet_to_img
    }
    for dir in DIR:
        for m in M:
            test_title = 'PreCIQ_m{}_{}_btpd'.format(m, dir)
            def ciq(img, **ciq_status):
                trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                # trans_img = img.copy()
                org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                # pre quantize
                status = {}
                pre_q, root, pre_groups = BTPD(S, m * 16, visualization=False)
                pre_mapped = mapping_pallet_to_img(trans_img, pre_q)
                print(f'{len(np.unique(org_S, axis=0))}')
                S = np.reshape(pre_mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                # only in case of sum
                print('pre quantize {} colors'.format(len(root.get_leaves())))
                q, root, groups = BTPD(S, m, visualization=False)
                # groupsから分割色の保存
                save_imgs = []
                pre_mapped = cv2.cvtColor(pre_mapped, cv2.COLOR_LAB2BGR)
                reshape_q = np.reshape(q, newshape=(m, 1, 3)).astype(np.uint8)
                retrans_q = cv2.cvtColor(reshape_q, cv2.COLOR_LAB2BGR)
                reshape_pre_q = np.reshape(pre_q, newshape=(len(pre_q), 1, 3)).astype(np.uint8)
                retrans_pre_q = cv2.cvtColor(reshape_pre_q, cv2.COLOR_LAB2BGR)
                save_imgs.extend(
                    [{'img': pre_mapped, 'filename': 'pre_mapped.jpg'}]
                )
                dict = {'palette': q,
                        'groups': [retrans_pre_q[:, 0, :], retrans_q[:, 0, :]],
                        'save_imgs': save_imgs}
                return dict
            CIQ_test(ciq, test_title, test_img=dir, **test_config)


def CIQ_test_PreQuantization_grad(M=[16], DIR=['sumple_img']):
    """
    重要度を，合計顕著度ではなく，その色の中で最大の顕著度とする
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    test_config = {
        'trans_flag': True,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': False,
        'save_tmp_imgs': True,
        'view_importance': False,
        'importance_eval': False,
        'ciq_error_eval': ciq_eval_set(),
        'mapping': mapping_pallet_to_img
    }
    for dir in DIR:
        for m in M:
            test_title = 'PreCIQ_m{}_{}_btpdgrad'.format(m, dir)
            def ciq(img, **ciq_status):
                trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                # trans_img = img.copy()
                org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                # pre quantize
                pre_q, root, pre_groups, ev_list = BTPD_LimitationSv(S, 500)
                pre_mapped = mapping_pallet_to_img(trans_img, pre_q)
                print(f'{len(np.unique(org_S, axis=0))}')
                S = np.reshape(pre_mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ev_list = ev_list[100:]
                num = 5
                ave = np.convolve(ev_list, np.ones(num) / float(num), 'valid')
                x = np.arange(0, len(ave), 1)
                ax.scatter(x, ave, marker='.', s=10)
                img_name = ciq_status['img_path']
                Path(f'{test_title}/{img_name}').mkdir(parents=True, exist_ok=True)
                plt.savefig(f'{test_title}/{img_name}/ex_list.jpg')
                # only in case of sum
                print('pre quantize {} colors'.format(len(root.get_leaves())))
                q, root, groups = BTPD(S, m, visualization=False)
                # groupsから分割色の保存
                save_imgs = []
                pre_mapped = cv2.cvtColor(pre_mapped, cv2.COLOR_LAB2BGR)
                reshape_q = np.reshape(q, newshape=(m, 1, 3)).astype(np.uint8)
                retrans_q = cv2.cvtColor(reshape_q, cv2.COLOR_LAB2BGR)
                reshape_pre_q = np.reshape(pre_q, newshape=(len(pre_q), 1, 3)).astype(np.uint8)
                retrans_pre_q = cv2.cvtColor(reshape_pre_q, cv2.COLOR_LAB2BGR)
                save_imgs.extend(
                    [{'img': pre_mapped, 'filename': 'pre_mapped.jpg'}]
                )
                dict = {'palette': q,
                        'groups': [retrans_pre_q[:, 0, :], retrans_q[:, 0, :]],
                        'save_imgs': save_imgs}
                return dict
            CIQ_test(ciq, test_title, test_img=dir, **test_config)


def CIQ_test_ProposalRemoveQuantile(M=[16], DIR=['sumple_img'], LIMIT=[3000]):
    """
    重要度を，合計顕著度ではなく，その色の中で最大の顕著度とする
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    test_config = {
        'trans_flag': True,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': True,
        'save_tmp_imgs': True,
        'view_importance': True,
        'importance_eval': get_importance_error_individually_color,
        'ciq_error_eval': ciq_eval_set(),
        'mapping': mapping_pallet_to_img
    }
    for dir in DIR:
        for m in M:
            for lim in LIMIT:
                test_title = 'ProposalSvSumWeight_m{}_{}_lim{}_LAB_frac'.format(m, dir, lim)

                def ciq(img, **ciq_status):
                    trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    # trans_img = img.copy()
                    org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                    org_Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map)
                    # pre quantize
                    pre_q, root, pre_groups = BTPD_WTSE_LimitationSv(S, org_Sv, lim)
                    pre_mapped = mapping_pallet_to_img(trans_img, pre_q)
                    print(f'{len(np.unique(org_S, axis=0))}')
                    # SM count in each colors
                    _, __, Sv_map = get_saliency_hist(pre_mapped, sm='SR')
                    Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map)
                    S = np.reshape(pre_mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                    uniq_S = np.unique(S, axis=0)
                    uniq_Sv = [Sv[np.where(np.all(color == S, axis=1))[0]] for color in uniq_S]
                    # len_max = np.max([len(sv_array) for sv_array in uniq_Sv])
                    # uniq_Sv = np.array([np.mean(sv_array) * (np.sum(sv_array) / len_max) for sv_array in uniq_Sv])
                    uniq_Sv = np.array([np.sum(sv_array) for sv_array in uniq_Sv])
                    uniq_Sv = uniq_Sv / np.max(uniq_Sv)
                    # only in case of sum
                    print('pre quantize {} colors'.format(len(root.get_leaves())))
                    q, root, groups = BTPD_WTSE(uniq_S, m, 1.0 / uniq_Sv)

                    # make colormap
                    indices = np.argsort(uniq_Sv)
                    # reshape_S = np.reshape(uniq_S, newshape=(len(indices), 1, 3))
                    retrans_S = cv2.cvtColor(uniq_S.astype(np.uint8), cv2.COLOR_LAB2BGR)
                    colormap = make_colormap(retrans_S[indices])

                    pre_mapped = cv2.cvtColor(pre_mapped, cv2.COLOR_LAB2BGR)
                    reshape_q = np.reshape(q, newshape=(m, 1, 3)).astype(np.uint8)
                    retrans_q = cv2.cvtColor(reshape_q, cv2.COLOR_LAB2BGR)
                    reshape_pre_q = np.reshape(pre_q, newshape=(len(pre_q), 1, 3)).astype(np.uint8)
                    retrans_pre_q = cv2.cvtColor(reshape_pre_q, cv2.COLOR_LAB2BGR)
                    dict = {'palette': q,
                            'groups': [retrans_pre_q[:, 0, :], retrans_q[:, 0, :]],
                            'save_imgs': [{'img': Sv_map, 'filename': 'tmp_Sv.jpg'},
                                          {'img': pre_mapped, 'filename': 'pre_mapped.jpg'},
                                          {'img': colormap, 'filename': 'colormap.jpg'}]}
                    return dict
                CIQ_test(ciq, test_title, test_img=dir, **test_config)


def CIQ_test_BTPD_WithImpoertance(M=[16], DIR=['sumple_img'], LIMIT=[3000]):
    """
    重要度を，合計顕著度ではなく，その色の中で最大の顕著度とする
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    for dir in DIR:
        for m in M:
            for lim in LIMIT:
                    code = cv2.COLOR_BGR2LAB
                    inverse_code = cv2.COLOR_LAB2BGR

                    def ciq(img):
                        trans_img = cv2.cvtColor(img, code)
                        # trans_img = img.copy()
                        org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                        Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map)

                        # pre quantize
                        q, root, pre_groups = BTPD_WTSE_LimitationSv(S, Sv, lim)
                        mapped = mapping_pallet_to_img(trans_img, q)
                        # SM count in each colors
                        _, __, Sv_map = get_saliency_hist(mapped, sm='SR')
                        Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map)
                        S = np.reshape(mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                        uniq_S = np.unique(S, axis=0)
                        uniq_Sv = np.array([np.sum(Sv[np.where(color == S)[0]])for color in uniq_S]).astype(np.float32)

                        # scaling Sv between 0 and 1 as weight of BTPD
                        uniq_Sv = (uniq_Sv / np.max(uniq_Sv)).astype(np.float32)
                        print('pre quantize {} colors'.format(len(root.get_leaves())))

                        q, root, groups = BTPD_WTSE(uniq_S, m, uniq_Sv)

                        # importances = uniq_S[np.argsort(uniq_Sv)]
                        dict = {'palette': q,
                                'groups': groups,
                                'tmp_sm': Sv_map}
                        return dict

                    SAVE = 'BTPD_WithImportance_m{}_{}_lim{}_LAB'.format(m, dir, lim)
                    CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code,
                             view_distribution=False, save_tmpSM=True, view_importance=False, importance_flag=False)


def CIQ_test_BTPD_SVcount_withoutPreQuantization(M=[16], DIR=['sumple_img'], LIMIT=[3000], DIV=[512]):
    """
    事前量子化なしで，合計顕著度を算出する
    :param M:
    :param DIR:
    :param PRE_Q:
    :param DIV:
    :return:
    """
    for dir in DIR:
        for m in M:
            for div in DIV:
                code = cv2.COLOR_BGR2LAB
                inverse_code = cv2.COLOR_LAB2BGR

                def ciq(img):
                    trans_img = cv2.cvtColor(img, code)
                    # SM count in each colors
                    _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                    S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                    Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32)
                    uniq_S = np.unique(S, axis=0)
                    uniq_Sv = np.round([np.mean(Sv[np.where(color == S)[0]] / div) for color in uniq_S]).astype(np.int)
                    tile_Sv = []
                    for color, sv in zip(uniq_S, uniq_Sv):
                        tile_Sv.extend(np.tile(color, (sv, 1)))
                    tile_Sv = np.array(tile_Sv)
                    tile_S = np.reshape(tile_Sv, newshape=(len(tile_Sv), 1, 3))
                    # W = (1.0 / (Sv + 1.0)).astype(np.float32)
                    W = Sv
                    q, root = BTPD(uniq_S, m)
                    leaves = root.get_leaves()
                    groups = []
                    for leaf in leaves:
                        index = leaf.get_data()['index']
                        pixels = tile_S[index]
                        pixels = np.reshape(pixels, newshape=(len(pixels), 3))
                        groups.append(pixels)
                    # importances = uniq_S[np.argsort(uniq_Sv)]
                    dict = {'palette': q,
                            'groups': groups,
                            'tmp_sm': Sv_map}
                    return dict
                SAVE = 'SVMean_m{}_{}_div{}'.format(m, dir, div)
                CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code,
                         view_distribution=True, save_tmpSM=True, view_importance=False)


if __name__ == '__main__':
    # CIQ_test_medianbased()
    # CIQ_test_sup5()
    # CIQ_test_sup1(M=[16, 32, 64], R=[0.1, 0.2])
    # CIQ_test_sup2()
    # CIQ_test_gradually()
    # CIQ_test_KMeans(M=[16, 32, 64], DIR=['sumple_img', 'misc'])
    # CIQ_test_BTPD_WithImpoertance(M=[32], DIR=['sumple_img'], LIMIT=[1000])
    # CIQ_test_ProposalTile(M=[16, 32], DIR=['sumple_img'], DIV=[1], LIMIT=[1000])
    # CIQ_test_BTPD_withSv(M=[16], DIR=['sumple_img'])
    # CIQ_test_ProposalSvSumWeight(M=[16, 32], DIR=['my_sumple'], LIMIT=[1000])
    # CIQ_test_BTPD_SVcount_withoutPreQuantization(M=[16, 32], DIR=['sumple_img'], DIV=[1, 4, 256])
    # CIQ_test_BTPD_MyPreQuantizeandSVcount(M=[16, 32], DIR=['sumple_img'], LIMIT=[3000], DIV=[32])
    # CIQ_test_BTPD_PreQuantizeandSVcount(M=[16, 32, 64], DIR=['sumple_img', 'misc'], PRE_Q=[128, 256, 512],
    #                                     DIV=[128, 256, 512])
    CIQ_test_PreQuantization_grad(M=[16], DIR=['sumple_img', 'sumple_org'])
    # CIQ_test_PreQuantization(M=[16, 32, 64], DIR=['sumple_img', 'sumple_org'])
    # CIQ_test_BTPD_PreQuantize(M=[16, 32], DIR=['sumple_img'], LIMIT=[3000, 4000], weighting=True)
    # CIQ_test_BTPD_PaletteDeterminationFromSv(M=[16, 32, 64], DIR=['sumple_img', 'misc'])
    # CIQ_test_BTPD_includingSv(M=[16, 32, 64], DIR=['sumple_img', 'misc'])
    # CIQ_test_BTPD_withSv(M=[16, 32], DIR=['sumple_img'])
    # CIQ_test_BTPD_InitializationSv(M=[16, 32, 64], DIR=['sumple_img'])
    # CIQ_test_BTPD_InitializationFromIncludingSv(M=[16, 32, 64], DIR=['sumple_img'])
