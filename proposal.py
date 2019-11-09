import numpy as np
import os
import cv2
import time
import pandas as pd
from sklearn.preprocessing import minmax_scale
from skimage.measure import compare_nrmse, compare_psnr, compare_ssim
from img_util import get_saliency_upper_th, make_colormap, get_saliency_hist, get_numcolors, get_spectralresidual, \
    get_saliency_lower_th, mapping_pallet_to_img, compare_labmse
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
    save_tmpSM = test_config['save_tmpSM']
    view_importance = test_config['view_importance']
    ciq_error_eval = test_config['ciq_error_eval']
    importance_eval = test_config['importance_eval']

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        root, ext = os.path.splitext(img_path)
        save_path = os.path.join(SAVE, root)
        img = cv2.imread(path)
        groups = None
        tmp_sm = None
        try:
            st = time.time()
            dict = ciq(img)
            en = time.time()
            palette = dict['palette']
            if view_distribution:
                groups = dict['groups']
            if save_tmpSM:
                tmp_sm = dict['tmp_sm']

        except KeyboardInterrupt:
            print('LinAlgError in {}'.format(img_path))
            continue

        if trans_flag:
            luv_img = cv2.cvtColor(img, code)
            mapped = mapping_pallet_to_img(luv_img, palette)
            mapped = np.reshape(mapped, newshape=img.shape)
            mapped = cv2.cvtColor(mapped, inverse_code)
        else:
            mapped = mapping_pallet_to_img(img, palette)
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
        imp_mse_rgb_1 = 0
        imp_mse_rgb_10 = 0
        if importance_eval:
            importance_img = get_img_importance(root)[:, :, 0]
            eval_errors = importance_eval(img, mapped, importance_img)
            for eval_error in eval_errors:
                score = eval_error['error']
                eval_index = eval_error['index']
                eval_array.append(score)
                eval_indices.append(eval_index)


            lin_imp_img = np.reshape(importance_img, newshape=(img.shape[0] * img.shape[1]))
            lin_imp_img = minmax_scale(lin_imp_img, (0, 255))

            for part in [.01, .05, .1]:
                top_dist_rgb = {'eval_func': lambda org, mapped: get_rgb_topdist(org, mapped, ratio=part),
                                'eval_index': f'Top Dist {part * 100}%'}
                top_dist_sv = {'eval_func': save_mse_in_eachSaliency,
                               'eval_index': f'Top Dist Sv {part * 100}%'}

                eval_list.extend([top_dist_rgb, top_dist_sv])
            thresh_1 = int(np.max(lin_imp_img) * 0.99)
            thresh_10 = int(np.max(lin_imp_img) * 0.9)

            index_1 = np.where(thresh_1 < lin_imp_img)
            index_10 = np.where(thresh_10 < lin_imp_img)

            imp_mse_rgb_1 = np.mean(rgb_mse[index_1])
            imp_mse_rgb_10 = np.mean(rgb_mse[index_10])

        df = pd.DataFrame([[img_path, en - st, *eval_array]], columns=['img_name', 'running time', *eval_indices])
        csv_path = os.path.join(SAVE, '{}_scores.csv'.format(test_name))

        if num != 0:
            pre_csv = pd.read_csv(csv_path, index_col=0)
            df = pre_csv.append(df)
        df.to_csv(csv_path)

        print('{} , by {}, calc time {}s'.format(img_path, test_name, en - st))
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

        # save tmp sm
        tmp_sm_path = os.path.join(save_path, 'tmp_sm_' + img_path)
        cv2.imwrite(tmp_sm_path, tmp_sm)

        if view_distribution:
            save_color_distribution(groups, save_path, 'Dist_' + root + '.jpg')

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


def save_color_distribution(groups, save_path, filename):
    width = 1
    while width ** 2 < len(groups):
        width += 1
    for num, group in enumerate(groups):
        COLUMNS = ['Variances', 'Variances by uniques', 'Mean']
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')  # 何もプロットしていないAxesでもAxisは自動的に作られる
        colors = group / 255.0
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], c=colors, marker=',', label='pixels')

        # Mean value
        mean = np.mean(group, axis=0).astype(np.uint8)
        colors = mean / 255.0
        ax.scatter(mean[0], mean[1], mean[2], c=[colors], marker='*', label='mean')

        # Values by uniques
        uniq = np.unique(group, axis=0)
        var_uniq = np.var(uniq)

        CellText = [[np.var(group), var_uniq, mean]]

        ax.set_title("{} Color Luminecence Distribution".format(num + 1))
        ax.set_xlabel("B")
        ax.set_ylabel("G")
        ax.set_zlabel("R")
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])
        # ax.legend()
        ax.grid()

        ax.table(cellText=CellText, colLabels=COLUMNS, loc='bottom')
        filepath = os.path.join(save_path, '{}_'.format(num) + filename)
        plt.savefig(filepath)
        plt.close()


def save_color_importanceerror(img, mapped, importance, save_path, filename):
    pass


def get_img_importance(img_path):
    SAVE = 'Importance_Map'
    path = os.path.join(SAVE, img_path)
    img_name = os.listdir(path)[0]
    path = os.path.join(path, img_name)
    imp_img = cv2.imread(path)
    return imp_img


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
        rgb_mse = np.linalg.norm(org - mapped, axis=2).reshape((org.shape[0] * org.shape[1])) / org.shape[2]
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
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2LAB
            inverse_code = cv2.COLOR_LAB2BGR

            def ciq(img):
                trans_img = cv2.cvtColor(img, code)
                S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint32)
                org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint32)
                _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float64)
                # Sv = 1.0 / (Sv + 1.0)
                # Sv = (255.0 - Sv) / 255.0
                Sv = Sv / 255.0
                q, root = BTPD_WTSE(S, m, Sv)
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
            SAVE = 'BTPD_WTSE_M{}_{}_LAB'.format(m, dir)
            CIQ_test(ciq, SAVE, test_img=dir, trans_flag=True, code=code, inverse_code=inverse_code,
                     view_distribution=True)


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


def CIQ_test_BTPD_MyPreQuantizeandOnlySV(M=[16], DIR=['sumple_img'], LIMIT=[3000], DIV=[1]):
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
        'save_tmpSM': True,
        'view_importance': True,
        'importance_flag': True,
        'ciq_error_eval': ciq_eval_set()
    }
    for dir in DIR:
        for m in M:
            for lim in LIMIT:
                for div in DIV:
                    def ciq(img):
                        trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        # trans_img = img.copy()
                        org_S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                        _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                        Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map)

                        # pre quantize
                        q, root, pre_groups = BTPD_WTSE_LimitationSv(S, Sv, lim)
                        mapped = mapping_pallet_to_img(trans_img, q)
                        print(f'{len(np.unique(org_S, axis=0))}')
                        # SM count in each colors
                        _, __, Sv_map = get_saliency_hist(mapped, sm='SR')
                        Sv = np.reshape(Sv_map, newshape=(len(S), 1, 1)).astype(np.float32) / np.max(Sv_map)
                        S = np.reshape(mapped, newshape=(len(S), 1, 3)).astype(np.uint64)
                        uniq_S = np.unique(S, axis=0)
                        uniq_Sv = np.round([np.sum(Sv[np.where(color == S)[0]]) / div for color in uniq_S]).astype(np.int)

                        # only in case of sum
                        uniq_Sv = (uniq_Sv / np.min(uniq_Sv)).astype(np.int)

                        tile_Sv = []
                        for color, sv in zip(uniq_S, uniq_Sv):
                            tile_Sv.extend(np.tile(color, (sv, 1)))
                        tile_Sv = np.array(tile_Sv)
                        tile_S = np.reshape(tile_Sv, newshape=(len(tile_Sv), 1, 3))
                        print('pre quantize {} colors'.format(len(root.get_leaves())))

                        q, root, groups = BTPD(tile_S, m)

                        # importances = uniq_S[np.argsort(uniq_Sv)]
                        dict = {'palette': q,
                                'groups': groups,
                                'tmp_sm': Sv_map}
                        return dict

                    SAVE = 'MyPreQuantizeSVSumW_m{}_{}_lim{}_div{}_LAB'.format(m, dir, lim, div)
                    CIQ_test(ciq, SAVE, test_img=dir, **test_config)


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
    CIQ_test_BTPD_MyPreQuantizeandOnlySV(M=[8], DIR=['sumple_img'], DIV=[1], LIMIT=[1000])
    # CIQ_test_BTPD_SVcount_withoutPreQuantization(M=[16, 32], DIR=['sumple_img'], DIV=[1, 4, 256])
    # CIQ_test_BTPD_MyPreQuantizeandSVcount(M=[16, 32], DIR=['sumple_img'], LIMIT=[3000], DIV=[32])
    # CIQ_test_BTPD_PreQuantizeandSVcount(M=[16, 32, 64], DIR=['sumple_img', 'misc'], PRE_Q=[128, 256, 512],
    #                                     DIV=[128, 256, 512])
    # CIQ_test_BTPD_PreQuantize(M=[16, 32], DIR=['sumple_img'], LIMIT=[3000, 4000], weighting=True)
    # CIQ_test_BTPD_PaletteDeterminationFromSv(M=[16, 32, 64], DIR=['sumple_img', 'misc'])
    # CIQ_test_BTPD_includingSv(M=[16, 32, 64], DIR=['sumple_img', 'misc'])
    # CIQ_test_BTPD_withSv(M=[16, 32], DIR=['sumple_img'])
    # CIQ_test_BTPD_InitializationSv(M=[16, 32, 64], DIR=['sumple_img'])
    # CIQ_test_BTPD_InitializationFromIncludingSv(M=[16, 32, 64], DIR=['sumple_img'])
