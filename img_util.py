import cv2
import os
import numpy as np
import pySaliencyMap
import matplotlib.pyplot as plt


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
    return saliencyMap


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


def get_saliency_hist(img):
    saliency_map = get_FineGrained(img)
    liner_sm = np.reshape(saliency_map, newshape=(img.shape[0] * img.shape[1]))
    hist, bins = np.histogram(liner_sm, bins=np.arange(0, 256, 1))
    return hist, bins, saliency_map


def test_saliency_map():
    DIR = 'sumple_org'
    SAVE = 'FineGrained'
    imgs = os.listdir(DIR)
    R = 0.2

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)
        pickup_sm = np.zeros(shape=img.shape)

        # saliency mapの保存
        saliency_map = get_FineGrained(img)
        save_path = os.path.join(SAVE, img_path)
        cv2.imwrite(save_path, (saliency_map * 1).astype(np.uint8))

        # histgramの保存
        liner_sm = np.reshape(saliency_map, newshape=(img.shape[0] * img.shape[1]))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(liner_sm, bins=256)
        ax.set_title('{} saliency map(FineGrained)'.format(img_path))
        ax.set_xlabel('luminescence')
        ax.set_ylabel('number')
        root, ext = os.path.splitext(img_path)
        save_fig = os.path.join(SAVE, 'SM_' + img_path.replace(ext, 'png'))
        plt.savefig(save_fig)

        # histgramの上位R%に属する画素のみ表示
        hist, bins = np.histogram(liner_sm, bins=np.arange(0, 256, 1))
        save_pickup = os.path.join(SAVE, 'pickup_' + img_path)
        th = int(R * len(liner_sm))
        assert len(liner_sm) == np.sum(hist), "liner_sum: {}, hist: {}".format(len(liner_sm), np.sum(hist))
        count = 0
        for num, bin in zip(hist[::-1], bins[::-1]):
            indices = np.where(saliency_map == int(bin))
            pickup_sm[indices] = img[indices]

            count += num
            if count >= th:
                break
        cv2.imwrite(save_pickup, pickup_sm)

        print('extract saliency map of img {} by use of {}'.format(save_path, SAVE))


def test_featureextraction():
    DIR = 'sumple_img'
    SAVE = 'AKAZE_POINTS'
    imgs = os.listdir(DIR)

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)
        img2 = draw_akazepoints(img)
        save_path = os.path.join(SAVE, img_path)
        cv2.imwrite(save_path, img2)
        print('save {}'.format(save_path))


if __name__ == '__main__':
    # bmp2jpg()
    test_saliency_map()
