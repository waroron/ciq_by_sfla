import cv2
import os
import numpy as np
import pySaliencyMap
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
from skimage.measure import compare_nrmse, compare_psnr
from sklearn.metrics import mean_squared_error


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image_cv):
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert('RGB')

    return image_pil


def get_surf_point(img):
    # 特徴抽出機の生成
    detector = cv2.xfeatures2d.SIFT_create()
    # kpは特徴的な点の位置 destは特徴を現すベクトル
    kp1, des1 = detector.detectAndCompute(img, None)
    return kp1, des1


def get_akaze_point(img):
    # 特徴抽出機の生成
    detector = cv2.AKAZE_create()
    # kpは特徴的な点の位置 destは特徴を現すベクトル
    kp1, des1 = detector.detectAndCompute(img, None)
    return kp1, des1


def draw_surfpoints(img):
    kp, des = get_surf_point(img)
    img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    return img2


def draw_akazepoints(img):
    kp, des = get_akaze_point(img)
    img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    return img2


def get_RARE_map(img):
    #RGB to YCbCr
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    # Gabor Filtering
    gaborConf = cv2.getGaborKernel((100, 100), 16.0, np.radians(0), 10, 0.5, 0)
    y_gabor = cv2.filter2D(y, -1, gaborConf)
    cr_gabor = cv2.filter2D(cr, -1, gaborConf)
    cb_gabor = cv2.filter2D(cb, -1, gaborConf)
    print('test')


def get_FineGrained(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    # versionによって255倍する必要あり
    return (saliencyMap * 255.0).astype(np.uint8)


def get_spectralresidual(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    # versionによって255倍する必要あり
    return (saliencyMap * 1.0).astype(np.uint8)


def get_saliency_map(img):
    sm = pySaliencyMap.pySaliencyMap(img.shape[1], img.shape[0])
    # computation
    saliency_map = sm.SMGetSM(img)
    return saliency_map


def bmp2jpg():
    DIR = 'sumple_img'
    SAVE = 'sumple_jpg'
    imgs = os.listdir(DIR)

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)
        save_path = os.path.join(SAVE, img_path.replace('bmp', 'jpg'))
        cv2.imwrite(save_path, img)
        print('save {}'.format(save_path))


def get_saliency_hist(img, sm='FineGrained'):
    if sm == 'FineGrained':
        saliency_map = get_FineGrained(img)
    else:
        saliency_map = get_spectralresidual(img)
    liner_sm = np.reshape(saliency_map, newshape=(img.shape[0] * img.shape[1]))
    hist, bins = np.histogram(liner_sm, bins=np.arange(0, 256, 1))
    return hist, bins, saliency_map


def compare_labmse(img1, img2):
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)

    return compare_nrmse(img1_lab, img2_lab)


def get_saliency_upper_th(img, R, sm='FineGrained'):
    hist, bins, sm = get_saliency_hist(img, sm=sm)
    th = int(R * np.sum(hist))
    count = 0
    extract = []
    extract_partition = []
    zeros = np.zeros(shape=img.shape)
    min_sm = 255
    for num, bin in zip(hist[::-1], bins[::-1]):
        indices = np.where(sm == int(bin))
        pixels = img[indices]
        zeros[indices] = img[indices]
        extract_partition.append(pixels)
        extract.extend(pixels)

        count += num
        if count >= th:
            min_sm = bin
            break
    return np.array(extract), np.array(extract_partition), zeros, min_sm


def get_saliency_lower_th(img, R, sm='FineGrained'):
    hist, bins, sm = get_saliency_hist(img, sm=sm)
    th = int(R * np.sum(hist))
    count = 0
    extract = []
    extract_partition = []
    zeros = np.zeros(shape=img.shape)
    for num, bin in zip(hist[::1], bins[::1]):
        indices = np.where(sm == int(bin))
        pixels = img[indices]
        zeros[indices] = img[indices]
        extract_partition.append(pixels)
        extract.extend(pixels)

        count += num
        if count >= th:
            break
    return np.array(extract), np.array(extract_partition), zeros


def mapping_pallet_to_img(img, pallete):
    dists = np.empty(shape=(img.shape[0], img.shape[1], len(pallete)))
    for num, pal in enumerate(pallete):
        dist = np.linalg.norm(img - pal, axis=2)
        dists[:, :, num] = dist

    pal = np.argmin(dists, axis=2)
    mapped_img = pallete[pal].astype(np.uint8)
    mapped_img = np.reshape(mapped_img, newshape=(img.shape[0], img.shape[1], 3))

    return mapped_img



def find_closest_color(pix, source):
    tile = np.tile(pix, len(source)).reshape(source.shape)
    dist = np.linalg.norm(tile - source, axis=2)
    index = np.argmin(dist)
    return source[index]



def FloydSteinbergDithering(img, palette):
    height, width = img.shape[0], img.shape[1]
    org = img.copy()
    for y in range(height):
        for x in range(width):
            closest = find_closest_color(org[y, x], palette)
            org[y, x] = closest
            q_err = org[y, x] - closest

            if x < width - 1:
                org[y, x + 1] = org[y, x + 1] + (7.0 * q_err) / 16
            if y < height - 1:
                org[y + 1, x] = org[y + 1, x] + (5.0 * q_err) / 16
                if x > 0:
                    org[y + 1, x - 1] = org[y + 1, x - 1] + (3.0 * q_err) / 16
                if x < width - 1:
                    org[y + 1, x + 1] = org[y + 1, x + 1] + q_err / 16.0
    return org


def make_colormap(colors, color_width=64):
    width_n_colors = int(np.sqrt(len(colors)) + 0.5)
    map_width = width_n_colors * color_width

    # カラーマップの生成
    color_map = np.zeros(shape=(map_width, map_width, 3))
    i, j = 0, 0
    for color in colors:
        x = int(i * color_width)
        y = int(j * color_width)
        color_map[y: y + color_width, x: x + color_width] = color

        if i >= map_width / color_width - 1:
            i = 0
            j += 1
        else:
            i += 1

    return color_map


def get_importancemap(img):
    # lin_img = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3))
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist, bins, sm = get_saliency_hist(lab_img, sm='SR')

    sum_imp_mat = np.zeros(shape=(img.shape[0], img.shape[1]))
    mean_imp_mat = np.zeros(shape=(img.shape[0], img.shape[1]))
    max_imp_mat = np.zeros(shape=(img.shape[0], img.shape[1]))
    all_colors = get_allcolors_from_img(img)
    # tmp = np.where(np.all(all_colors[0] == img, axis=2))
    # pix = img[tmp]
    colors_sv_array = [sm[np.where(np.all(color == img, axis=2))] for color in all_colors]
    sum_importance = np.array([np.sum(sm_array) for sm_array in colors_sv_array])
    mean_importance = np.array([np.mean(sm_array) for sm_array in colors_sv_array])
    max_importance = np.array([np.max(sm_array) for sm_array in colors_sv_array])

    for n, color in enumerate(all_colors):
        index = np.where(np.all(color == img, axis=2))
        sum_imp_mat[index[:2]] = sum_importance[n]
        mean_imp_mat[index[:2]] = mean_importance[n]
        max_imp_mat[index[:2]] = max_importance[n]

    return sum_imp_mat, mean_imp_mat, max_imp_mat


def get_allcolors_from_img(img):
    flattened_img = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint8)
    # flattened_img[1] = flattened_img[0]
    uniq_S = np.unique(flattened_img, axis=0)
    return uniq_S


def get_numcolors(img):
    if len(img.shape) > 2:
        img = np.reshape(img, newshape=(img.shape[0] * img.shape[1], img.shape[2]))
    df = pd.DataFrame(img)
    return len(df.drop_duplicates().values)


def test_saliency_map():
    DIR = 'sumple_img'
    SAVE = 'SM_map_spectralresidual_LAB'
    imgs = os.listdir(DIR)
    R = np.arange(1.0, 0, -0.1)
    PART = 8
    plt.rcParams["font.size"] = 14

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        org_img = cv2.imread(path)
        img = org_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        root, ext = os.path.splitext(img_path)
        img_dir = os.path.join(SAVE, root)

        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        # saliency mapの保存
        saliency_map = get_spectralresidual(img)
        liner_sm = np.reshape(saliency_map, newshape=(img.shape[0] * img.shape[1]))
        save_path = os.path.join(img_dir, f'Smap_{root}{ext}')
        cv2.imwrite(save_path, (saliency_map * 1).astype(np.uint8))
        print('save saliency map as img {}'.format(save_path))

        for n in range(len(R)):
            # histgramの上位R%に属する画素のみ表示
            pickup_sm = np.zeros(shape=img.shape)
            hist, bins = np.histogram(liner_sm, bins=np.arange(0, 256, 1))
            save_pickup = os.path.join(img_dir, 'pickup_upper{:.2}'.format(R[n]) + img_path)
            th = int(R[n] * len(liner_sm))
            assert len(liner_sm) == np.sum(hist), "liner_sum: {}, hist: {}".format(len(liner_sm), np.sum(hist))
            count = 0
            for num, bin in zip(hist[::-1], bins[::-1]):
                indices = np.where(saliency_map == int(bin))
                pickup_sm[indices] = org_img[indices]

                count += num
                if count >= th:
                    break
            cv2.imwrite(save_pickup, pickup_sm)
            print('extract saliency map of img {} by use of {}'.format(save_pickup, img_dir))

        for n in range(len(R)):
            # histgramの上位R%に属する画素のみ表示
            pickup_sm = np.zeros(shape=img.shape)
            hist, bins = np.histogram(liner_sm, bins=np.arange(0, 256, 1))
            save_pickup = os.path.join(img_dir, 'pickup_lower{:.2}'.format(R[n]) + img_path)
            th = int(R[n] * len(liner_sm))
            assert len(liner_sm) == np.sum(hist), "liner_sum: {}, hist: {}".format(len(liner_sm), np.sum(hist))
            count = 0
            for num, bin in zip(hist[::1], bins[::1]):
                indices = np.where(saliency_map == int(bin))
                pickup_sm[indices] = org_img[indices]

                count += num
                if count >= th:
                    break
            cv2.imwrite(save_pickup, pickup_sm)
            print('extract saliency map of img {} by use of {}'.format(save_pickup, img_dir))

        # histgramの保存
        liner_sm = np.reshape(saliency_map, newshape=(img.shape[0] * img.shape[1]))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(liner_sm, bins=256)
        ax.set_title('{} saliency map'.format(img_path))
        ax.set_xlabel('saliency')
        ax.set_ylabel('# of pixels')
        save_fig = os.path.join(img_dir, 'SM_' + img_path.replace(ext, '.jpg'))
        plt.tight_layout()
        plt.savefig(save_fig)
        print('save histogram as img {}'.format(save_fig))

        # histgramのbinsをPARTずつ区切ってそれぞれの画素を表示
        for bin in range(0, 256, PART):
            pickup_sm = np.zeros(shape=img.shape)
            save_pickup = os.path.join(img_dir, 'pickup_bin{}-bin{}'.format(bin, bin + PART) + img_path)
            for n in range(bin, bin + PART):
                indices = np.where(saliency_map == int(n))
                pickup_sm[indices] = org_img[indices]
            cv2.imwrite(save_pickup, pickup_sm)
            print('save fig {}'.format(save_pickup))


def test_importance_map():
    DIR = 'sumple_img'
    SAVE = 'Importance_Map'
    imgs = os.listdir(DIR)

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        org_img = cv2.imread(path)
        img = org_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        root, ext = os.path.splitext(img_path)
        img_dir = os.path.join(SAVE, root)

        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        # importance mapの保存
        importance_map = get_importancemap(img)
        save_path = os.path.join(img_dir, f'{root}.csv')
        df = pd.DataFrame(importance_map)
        df.to_csv(save_path)
        print('save Importance_Map map as img {}'.format(save_path))


def get_all_preimportance_map():
    DIR = 'ProposalSvSumWeight_m32_sumple_img_lim1000_LAB_sum'
    SAVE = 'Importance_Map'
    imgs = os.listdir(DIR)

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        org_img = cv2.imread(f'{path}/pre_mapped.jpg')
        img = org_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        root, ext = os.path.splitext(img_path)
        img_dir = os.path.join(SAVE, root)

        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        # importance mapの保存
        sum_imp, mean_imp, max_imp = get_importancemap(img)
        labels = ['sum_imp', 'mean_imp', 'max_imp']
        for imp_mat, label in zip([sum_imp, mean_imp, max_imp], labels):
            save_path = os.path.join(img_dir, f'premapped_{label}_importance.csv')
            df = pd.DataFrame(imp_mat)
            df.to_csv(save_path)
            print('save Importance_Map map as img {}'.format(save_path))


def test_sum_saluency():
    DIR = 'sumple_img'
    SAVE = 'SM_map'
    imgs = os.listdir(DIR)
    R = 0.25

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)

        # saliency mapの保存
        saliency_map = get_spectralresidual(img)

        # 各色のヒストグラム
        pix_vec = get_saliency_upper_th(img, R)

        hist = np.zeros(shape=(256, 256, 256))
        for pix in pix_vec:
            hist[pix] += 1
        print(hist)

        # pix_vec = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 3))
        colors = pd.DataFrame(pix_vec)
        count = colors.duplicated().value_counts()
        for color in colors:
            hist[color] = len(np.where(pix_vec == color))
        print('test')


def test_smextraction():
    DIR = 'sumple_img'
    SAVE = 'SM_map'
    imgs = os.listdir(DIR)
    SELECT = 2048 * 2

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)
        extract, _ = get_saliency_upper_th(img, 1)
        extract = pd.DataFrame(extract)
        # 顕著度TOP256色を表示
        top = extract[:SELECT]
        color_map = make_colormap(top, width=SELECT * 4)
        save_path = os.path.join(SAVE, 'TOP{}_'.format(SELECT) + img_path)
        cv2.imwrite(save_path, color_map)
        print('save {}'.format(save_path))


def test_sm_variance():
    DIR = 'sumple_img'
    SAVE = 'SM_map_spectralresidual'
    imgs = os.listdir(DIR)
    PARTITION = 1

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hist, bins, sm = get_saliency_hist(img)
        root, ext = os.path.splitext(img_path)
        img_dir = os.path.join(SAVE, root)

        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)

        vars = []
        xlabel = []
        for bin in bins[::-PARTITION]:
            parted_extract = []
            for n in range(PARTITION):
                indices = np.where(sm == int(n + bin))
                pixels = img[indices]
                parted_extract.extend(pixels)

            parted_extract = np.array(parted_extract)
            var = parted_extract.var()
            vars.append(var)
            xlabel.append(bin)

        # histgramの保存
        vars = np.array(vars)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(xlabel, vars)
        ax.set_title('{} saliency variance(spectralresidual)'.format(img_path))
        ax.set_xlabel('saliency')
        ax.set_ylabel('variance')

        save_fig = os.path.join(img_dir, 'Var_hist_' + img_path.replace(ext, 'png'))
        plt.savefig(save_fig)

        print('save {}'.format(save_fig))


if __name__ == '__main__':
    # bmp2jpg()
    # test_sum_saluency()
    # test_smextraction()
    # test_importance_map()
    # get_all_preimportance_map()
    # test_importance_map()
    test_saliency_map()
    # test_sm_variance()
