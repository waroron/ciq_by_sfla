import numpy as np
import os
import cv2
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from skimage.measure import compare_nrmse, compare_psnr
from img_util import get_saliency_upper_th, make_colormap


def SFLA(fit, create_frog, n_frogs=20, n_mem=5, T_max=100, J_max=5, rho=0.5):
    perm = np.arange(0, n_frogs, 1)
    init_frogs = np.array([create_frog() for _ in range(n_frogs)])
    hist_frogs = []
    hist_bestfrogs = []

    def eval_frogs(i_frogs):
        score = np.zeros((len(i_frogs)))
        for j in range((len(i_frogs))):
            score[j] = fit(i_frogs[j])

        return score

    shuffled_frogs = init_frogs.copy()
    hist_frogs.append(shuffled_frogs)
    global_best_index = 0
    for t in range(T_max):
        shuffled_index = np.random.permutation(perm)
        shuffled_frogs = shuffled_frogs[shuffled_index]
        shuffled_fitness = eval_frogs(shuffled_frogs)
        global_best_index = np.argmax(shuffled_fitness)
        hist_bestfrogs.append(shuffled_frogs[global_best_index])

        print('{} gens:  best score: {}'.format(t, shuffled_fitness[global_best_index]))

        # divide all frogs into several memeplexes
        assert n_frogs % n_mem == 0, "wrong setting of num_frogs or num_memeplex"

        for bd in range(n_mem, n_frogs, n_mem):
            for j in range(J_max):
                index_bius = bd - n_mem
                mem_best_index = np.argmax(shuffled_fitness[index_bius: bd])
                mem_worst_index = np.argmin(shuffled_fitness[index_bius: bd])
                # compute Eqs.(1) and (2)
                D = rho * (shuffled_frogs[mem_best_index + index_bius] - shuffled_frogs[mem_worst_index + index_bius])
                next_x_worst = shuffled_frogs[mem_worst_index + index_bius] + D
                next_x_worst_fit = fit(next_x_worst)

                if next_x_worst_fit < shuffled_fitness[mem_worst_index + index_bius]:
                    # apply Eqs.(1) and (3)
                    D = rho * (shuffled_frogs[global_best_index] - shuffled_frogs[mem_worst_index + bd])
                    next_x_worst = shuffled_frogs[mem_worst_index + bd] + D
                    next_x_worst_fit = fit(next_x_worst)

                    if next_x_worst_fit < shuffled_fitness[mem_worst_index + index_bius]:
                        # Move the worst frog to a random position
                        next_x_worst = create_frog()
                shuffled_frogs[mem_worst_index + index_bius] = next_x_worst
                shuffled_fitness[mem_worst_index + index_bius] = next_x_worst_fit

    return shuffled_frogs[global_best_index], hist_bestfrogs, hist_frogs


def BTPD(S, M):
    C = []
    R = []
    m = []
    N = []
    q = []

    def get_R(c):
        sum = (c[0] * c[0].T).copy()
        for s in c[1:]:
            tmp = s * s.T
            sum += tmp
        return sum

    def get_m(c):
        sum = c[0].copy()
        for s in c[1:]:
            sum += s
        return sum

    def get_N(c):
        return len(c)

    C.append(S)
    R.append(get_R(C[0]))
    m.append(get_m(C[0]))
    N.append(get_N(C[0]))
    q.append(m[0] / N[0])

    for num in range(M - 1):
        R_ = R[num] - (np.dot(m[num], m[num].T) / N[num])
        W, v = np.linalg.eig(R_)
        e = v[np.argmax(W)]

        criteria = np.dot(e, q[num][0])
        compare = np.dot(e, C[num][:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n <= 0 or num_c2n1 <= 0:
            raise ValueError("The target image doesn't have many colors")
            # print("the class should be divided into 2 classes, num: {}: len1: {}, len2: {}".format(num, num_c2n, num_c2n1))
            # print('var: {}'.format(np.var(C[num], axis=0)))
            # for n, c in enumerate(C):
            #     print('{} th C: len {}'.format(n + 1, len(c)))
            # print('len compare: {}'.format(len(compare)))
        # assert num_c2n > 0 and num_c2n1 > 0, "the class should be divided into 2 classes, num: {}: len1: {}, len2: {}".format(num, num_c2n, num_c2n1)

        c_2n = np.reshape(C[num][c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(C[num][c_2n1_index[0]], (num_c2n1, 1, 3))

        C.append(c_2n)
        C.append(c_2n1)

        R.append(get_R(c_2n))
        m.append(get_m(c_2n))
        N.append(get_N(c_2n))
        q.append(m[-1] / N[-1])

        R.append(R[num] - R[-1])
        m.append(m[num] - m[-1])
        N.append(N[num] - N[-1])
        q.append(m[-1] / N[-1])

    color_palette = np.round(q[len(q) - M:])

    return color_palette


def BTPD_WTSE(S, M, h):
    C = []
    R = []
    m = []
    N = []
    q = []
    W = []

    y_weight = np.array([0.300, 0.586, 0.115])  # RGBの順番
    y = y_weight * S

    for _y in y:
        w_s = np.power(1.0 / (h * (np.min(np.linalg.norm(_y, ord=2), 16) + 2.0)), 2.0)
        W.append(w_s)

    def get_R(c):
        sum = (W[0] * c[0] * c[0].T).copy()
        for w, s in zip(W[1:], c[1:]):
            tmp = w * s * s.T
            sum += tmp
        return sum

    def get_m(c):
        sum = W[0] * c[0].copy()
        for w, s in zip(W[1:], c[1:]):
            sum += w * s
        return sum

    def get_N():
        sum = W[0].copy()
        for w in W[1:]:
            sum += w
        return sum


    C.append(S)
    R.append(get_R(C[0]))
    m.append(get_m(C[0]))
    N.append(get_N())
    q.append(m[0] / N[0])

    for num in range(M - 1):
        R_ = R[num] - (m[num] * m[num].T) / N[num]
        W, v = np.linalg.eig(R_)
        e = v[np.argmax(W)]

        criteria = np.dot(e, q[num][0])
        compare = np.dot(e, C[num][:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        c_2n = np.reshape(C[num][c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(C[num][c_2n1_index[0]], (num_c2n1, 1, 3))

        C.append(c_2n)
        C.append(c_2n1)

        R.append(get_R(c_2n))
        m.append(get_m(c_2n))
        N.append(get_N(c_2n))
        q.append(m[-1] / N[-1])

        R.append(R[num] - R[-1])
        m.append(m[num] - m[-1])
        N.append(N[num] - N[-1])
        q.append(m[-1] / N[-1])

    color_palette = np.round(q[len(q) - M:])
    return color_palette


def EBW_CIQ(S, M):
    C = []
    R = []
    m = []
    N = []
    q = []
    M_0 = 0

    def get_R(c):
        sum = (c[0] * c[0].T).copy()
        for s in c[1:]:
            tmp = s * s.T
            sum += tmp
        return sum

    def get_m(c):
        sum = c[0].copy()
        for s in c[1:]:
            sum += s
        return sum

    def get_N(c):
        return len(c)


    C.append(S)
    R.append(get_R(C[0]))
    m.append(get_m(C[0]))
    N.append(get_N(C[0]))
    q.append(m[0] / N[0])

    for num in range(M - 1):
        R_ = R[num] - (m[num] * m[num].T) / N[num]
        W, v = np.linalg.eig(R_)
        e = v[np.argmax(W)]

        criteria = np.dot(e, q[num][0])
        compare = np.dot(e, C[num][:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        c_2n = np.reshape(C[num][c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(C[num][c_2n1_index[0]], (num_c2n1, 1, 3))

        C.append(c_2n)
        C.append(c_2n1)

        R.append(get_R(c_2n))
        m.append(get_m(c_2n))
        N.append(get_N(c_2n))
        q.append(m[-1] / N[-1])

        R.append(R[num] - R[-1])
        m.append(m[num] - m[-1])
        N.append(N[num] - N[-1])
        q.append(m[-1] / N[-1])

    color_palette = np.round(q[len(q) - M:])
    return color_palette


def Wu_method(S, K):
    # initialization
    # numpy使って共分散行列を算出した後，最大固有値の固有ベクトルからprojectionしようと思ったが
    # MemoryError吐くのでPCAクラスを直接使った．PCAを用いた際の動きが，論文中に詳しく記載されていなかったが
    # 数式を見るに，おそらくPCAを使って全画素のデータを1次元に変換したときと等価だと思われる．
    pca = PCA(n_components=1)
    pca.fit(S[:, 0, :])
    # C = np.cov(S)
    # W, _v = np.linalg.eig(C)
    # v = _v[np.argmax(W)]
    # v = pca.components_
    mapping = np.reshape(pca.transform(S[:, 0, :]), len(S))
    # N projections into M < N buckets
    M = 128
    sumpled_mapping = mapping[::M]
    print('{} --> {}'.format(len(S), len(sumpled_mapping)))
    N = len(sumpled_mapping)
    E = np.empty(shape=N - 1)
    L = np.empty(shape=(K, N))
    listed_S = np.empty(shape=S.shape)

    sorted_index = np.argsort(mapping)
    listed_S = S[sorted_index]
    listed_mapping = mapping[sorted_index]

    def Lchain(k, n):
        t = n - 1
        q = np.empty(shape=K + 1)
        for j in range(k - 1, -1, -1):
            q[j] = L[j, t]
            t = int(L[j, t])
            print(t)
        return q

    def W0(n):
        return len(listed_S[:n])

    def W1(d, n):
        cd = 0
        for c in listed_S[:n]:
            cd += c[:, d]
        return cd

    def W2(n):
        cd = 0
        for c in listed_S[:n]:
            cd += np.sum(c) ** 2
        return cd

    print('start preconputing')
    # precompute for evaluating error
    w0 = np.empty(shape=N)
    w1 = np.empty(shape=(3, N))
    w2 = np.empty(shape=N)

    for n in range(N):
        w0[n] = W0(n + 1)
        w2[n] = W2(n + 1)
        print(n)
        for d in range(2):
            w1[d, n] = W1(d, n + 1)

    def quantized_error(a, b):
        """
        z = np.median(listed_S[a:b - 1])
        err = 0
        for c in listed_S[a:b - 1]:
            err += np.linalg.norm(c - z)
        return err
        だとめちゃくちゃ遅いので，Section4.4よりlinear timeで計算するために工夫する
        """
        frac = np.sum((w1[:, b] - w1[:, a]) ** 2) / (w0[b] - w0[a])
        return w2[b] - w2[a] - frac

    for num in range(1, N):
        # 実際はe(0, 0)は存在しないため(0, 0) --> (0, 1)と置き換えている
        E[num - 1] = quantized_error(0, num)

    for num in range(0, K):
        L[num, num] = num - 1

    print('initialization over')
    for k in range(2, K + 1):
        print(k)
        for n in range(k + 1, N - K + k + 1):
            cut = n - 1
            e = E[n - 2]
            for t in range(n - 2, k - 2, -1):
                q_err = quantized_error(t, n - 1)
                if E[t] + q_err < e:
                    cut = t
                    e = E[t] + q_err
            L[k - 1, n - 1] = cut
            E[n - 2] = e
    return Lchain(K, N)


def OneMaxBySFLA():
    def fit(frog):
        overflow_index = np.where(frog >= 0.5)
        underflow_index = np.where(frog < 0.5)
        frog[overflow_index[0]] = 1.0
        frog[underflow_index[0]] = 0.0
        return np.sum(frog)

    def create_frog():
        return np.random.randint(0, 2, 10).astype(np.float32)

    SFLA(fit, create_frog)


def BTPD_CIQ(img, M):
    """
    Binary Splittingによってカラーパレットを設計する．
    マッピング部も様々なアルゴリズムが提案されているため，カラーパレットとマッピング部で分割している．
    :param img: OpenCVによって得た入力画像
    :param M: 量子化数，カラーパレットサイズ
    :return:
    """
    S = np.reshape(img, (img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
    color_palette = BTPD(S, M)

    return color_palette


def Wu_CIQ(img, M):
    S = np.reshape(img, (img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
    color_palette = Wu_method(S, M)

    return color_palette


def KMeans_CIQ(S, K, n_iterations=None, init_array=None):
    """
    K-MeansによるCIQ
    初期値と最大イテレーション数を与えられたときのみ(たぶんPSO-CIQとかCQ-ABCを適用するときとか)
    それらを引数として与えて処理する．
    :param S: 入力画像を，(画素数, 1, 3)とreshapeしている
    :param K:
    :param n_iterations:
    :param init_array:
    :return:
    """
    if n_iterations:
        kmeans = KMeans(n_clusters=K,
                        init=init_array,
                        max_iter=n_iterations)

    else:
        kmeans = KMeans(n_clusters=K)

    features = np.reshape(S, (len(S), 3))
    # X_new = kmeans.fit_transform(features)
    kmeans.fit(features)
    X_new = kmeans.cluster_centers_[kmeans.labels_]

    return kmeans, X_new


def CQ_ABC(img, K):
    pass


def CIEDE76(img1, img2):
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)

    # Cab_1 = np.sqrt(img1_lab[1] ** 2 + img1_lab[2] ** 2)
    # Cab_2 = np.sqrt(img2_lab[1] ** 2 + img2_lab[2] ** 2)
    #
    # hab_1 = np.arctan(img1_lab[2] / img1_lab[1])
    # hab_2 = np.arctan(img2_lab[2] / img2_lab[1])

    dist = np.linalg.norm(img1_lab - img2_lab)
    return dist


def PSO_CIQ(img, K, n_particles, t_max, p_kmeans, kmeans_iteration, w, c1, c2):
    S = np.reshape(img, (img.shape[0] * img.shape[1], 3)).astype(np.uint64)
    Z = np.empty(shape=(img.shape[0] * img.shape[1], 3))
    Fitness = np.empty(shape=(n_particles, 1))
    best_particle = np.empty(shape=(t_max, K, 3))

    def fit(S, Z):
        # 論文とは違う評価方法になってるが，たぶん論文の方は記述ミス
        # 要確認
        return mean_squared_error(S, Z)

    def update_particle(x, y, y_):
        # 論文内の数式によると，更新する粒子がベクトルで表現されているが，実際には(K, 3)の行列であるため，
        # 各定数の扱いがわからんので，とりあえず各定数も(K, 3)の行列にする．
        r1 = np.random.uniform(0, 1, (K, 3))
        r2 = np.random.uniform(0, 1, (K, 3))
        v = w * y + c1 * r1 * (y - x) + c2 * r2 * (y_ - x)
        return x + v

    # 1. initialization
    particles = np.random.randint(0, 256, size=(n_particles, K, 3))

    # 2. For t = 1 to to_max
    for t in range(t_max):
        print(t)
        # (a) For each particle j
        for num, particle in enumerate(particles):
            this_p = np.random.uniform(0, 1)

            if this_p < p_kmeans:
                # Apply K-means for a few iterations
                kmeans, Z = KMeans_CIQ(S, K, kmeans_iteration, particle)

            # calculate fitness
            Fitness[num] = fit(S, Z)
        # Find the global best solution
        best_index = np.argmin(Fitness)
        best_particle[t] = particles[best_index]
        global_best_index = np.argmin(Fitness)

        # Update the centroids
        for num in range(n_particles):
            particles[num] = update_particle(particles[num], particles[best_index], best_particle[global_best_index])

    return particles


def compare_labmse(img1, img2):
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)

    return compare_nrmse(img1_lab, img2_lab)


def CIQ_test(ciq, test_name, test_img='sumple_img'):
    DIR = test_img
    SAVE = test_name
    imgs = os.listdir(DIR)
    INDICES = ['img_name', 'NRMSE', 'PSNR', 'Lab_NRMSE']

    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)

    for num, img_path in enumerate(imgs):
        path = os.path.join(DIR, img_path)
        save_path = os.path.join(SAVE, img_path)
        img = cv2.imread(path)
        st = time.time()
        try:
            palette = ciq(img)
        except np.linalg.LinAlgError:
            print('LinAlgError in {}'.format(img_path))
            continue
        except ValueError as me:
            print(me)
            print('Error in : {}'.format(img_path))
            continue

        en = time.time()
        mapped = mapping_pallet_to_img(img, palette)
        mapped = np.reshape(mapped, newshape=img.shape)

        # eval
        nrmse = compare_nrmse(img, mapped)
        psnr = compare_psnr(img, mapped)
        lab_nrmse = compare_labmse(img, mapped)

        df = pd.DataFrame([[img_path, nrmse, psnr, lab_nrmse]], columns=INDICES)
        csv_path = os.path.join(SAVE, '{}_scores.csv'.format(test_name))

        if num != 0:
            pre_csv = pd.read_csv(csv_path, index_col=0)
            df = pre_csv.append(df)
        df.to_csv(csv_path)

        print('{} , by {}, calc time {}s'.format(img_path, test_name, en - st))
        # mapped = mapping_pallet_to_img(img, q)
        cv2.imwrite(save_path, mapped)

        # save color map
        save_path = os.path.join(SAVE, 'map_' + img_path)
        color_map = make_colormap(palette)
        cv2.imwrite(save_path, color_map)


def CIQ_test_BTPD():
    DIR = 'sumple_img'
    SAVE = 'BTPD'
    M = 16

    def ciq(img):
        q = BTPD_CIQ(img, M)
        return q

    CIQ_test(ciq, SAVE)


def CIQ_test_PSO():
    TEST_NAME = 'PSO_CIQ'
    DIR = 'sumple_img'
    K = 16
    t_max = 100
    n_p = 50
    p_kmeans = 0.5
    kmeans_iteration = 10
    w = 0.729
    c1 = 1.4955
    c2 = 1.4955

    def ciq(img):
        palette = PSO_CIQ(img, K, n_p, t_max, p_kmeans, kmeans_iteration, w, c1, c2)
        return palette

    CIQ_test(ciq, TEST_NAME, DIR)


def CIQ_test_Wu():
    DIR = 'sumple_img'
    M = 16
    imgs = os.listdir(DIR)

    for img_path in imgs:
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)
        st = time.time()
        q = Wu_CIQ(img, M)
        en = time.time()
        print('{}: {} colors pallete by Wu\'s method: time {}'.format(img_path, M, en - st))
        for pix in q:
            print(pix)

        print('\n\n')


def CIQ_test_KMeans():
    DIR = 'sumple_img'
    SAVE = 'KMeans'
    M = 16

    def ciq(img):
        S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], img.shape[2]))
        kmeans, q = KMeans_CIQ(S, M)
        return kmeans.cluster_centers_

    CIQ_test(ciq, SAVE, DIR)


def CIQ_test_SFLA():
    K = 16

    def create_color_palette():
        return np.random.randint(0, 256, size=(K, 3))

    DIR = 'sumple_img'
    SAVE = 'SFLA_LabPSNR'

    def ciq(img):
        def psnr(frog):
            mapped = mapping_pallet_to_img(img, frog)
            psnr = compare_psnr(img, mapped)
            return psnr

        def Lab_psnr(frog):
            mapped = mapping_pallet_to_img(img, frog)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            mapped_lab = cv2.cvtColor(mapped, cv2.COLOR_BGR2Lab)
            psnr = compare_psnr(img_lab, mapped_lab)
            return psnr

        frog, _, __ = SFLA(Lab_psnr, create_color_palette)
        return frog

    CIQ_test(ciq, SAVE, DIR)


def CIQ_test_besed_on_SM():
    M = 16
    R = 0.5
    DIR = 'sumple_img'
    SAVE = 'KMeans_SMbased_R{:.2g}'.format(R)

    def ciq(img):
        extract, _ = get_saliency_upper_th(img, R)
        S = np.reshape(extract, newshape=(len(extract), img.shape[2]))
        kmeans, q = KMeans_CIQ(S, M)
        q = kmeans.cluster_centers_
        return q

    CIQ_test(ciq, SAVE, DIR)


def CIQ_test_gradually():
    DIR = 'sumple_img'
    SAVE = 'BTPD_1024_16'
    M = [1024, 16]

    def ciq(img):
        S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 1, img.shape[2]))
        for m in M:
            S = np.reshape(S, newshape=(len(S), 1, 3)).astype(np.uint64)
            print('len S: {}'.format(len(S)))
            q = BTPD(S, m)
            S = q
        return q

    CIQ_test(ciq, SAVE, DIR)


def CIQ_test_sup1():
    # 少数色でも顕著度が高ければ保存する
    # --> 顕著度ヒストグラムが一様になるように高顕著度の色を増やしてCIQ
    M = 16
    R = 0.05
    P = 10
    DIR = 'sumple_img'
    SAVE = 'KMeans_Sup1_R{:.2g}'.format(R)

    alpha = 0
    beta = 0
    def get_priority(sv, n):
            return alpha * sv + beta * n

    def ciq(img):
        extract, parted_extract = get_saliency_upper_th(img, R)
        colors = pd.DataFrame(extract).drop_duplicates().values
        print('{} colors exist'.format(len(colors)))
        hist = np.zeros(shape=len(colors))
        for num in range(len(colors)):
            index = np.where((img == colors[num]).all(axis=2))
            hist[num] = len(index[0])
        indices = np.argmax(hist)


        kmeans, _ = BTPD(extract, M)
        return kmeans.cluster_centers_

    CIQ_test(ciq, SAVE, DIR)


def CIQ_test_suq2():
    # 顕著度を何分割化し(高顕著度であれば小さく分割し，低顕著度であれば大きく分割する)，
    # 各分割に属する色で，何色かに量子化する
    pass


def CIQ_test_sup3():
    # 少数色でも顕著度が高ければ保存する
    # --> R = 0 - 0.1 の画素数をR = 0.4 - 0.5の画素数ほどに増加させる
    M = 16
    R = 0.5
    DIR = 'sumple_img'
    SAVE = 'KMeans_Sup1_R{:.2g}'.format(R)

    def ciq(img):
        extract = get_saliency_upper_th(img, R)
        partition = int(len(extract) / (R * 10)) + 1
        parted_extract = []

        for num in range(0, len(extract), partition):
            parted_extract.append(extract[num: num + partition])

        S = []
        l = len(parted_extract)
        for num in range(len(parted_extract)):
            n = len(parted_extract[num])
            th = len(parted_extract[l - 1 - num])
            count = 0
            if n <= 0:
                continue
            S.extend(extract[num])
            count += n
            while count < th:
                S.extend(extract[num])
                count += n

        S = np.reshape(np.array(S), newshape=(len(S), img.shape[2]))
        kmeans, q = KMeans_CIQ(S, M)
        return kmeans.cluster_centers_

    CIQ_test(ciq, SAVE, DIR)


def mapping_pallet_to_img(img, pallete):
    dists = np.empty(shape=(img.shape[0], img.shape[1], len(pallete)))
    for num, pal in enumerate(pallete):
        dist = np.linalg.norm(img - pal, axis=2)
        dists[:, :, num] = dist

    pal = np.argmin(dists, axis=2)
    mapped_img = pallete[pal].astype(np.uint8)

    return mapped_img


if __name__ == '__main__':
    CIQ_test_gradually()
    # CIQ_test_BTPD()
    # CIQ_test_sup1()
    # CIQ_test_besed_on_SM()
    # CIQ_test_KMeans()
    # CIQ_test_PSO()
    # OneMaxBySFLA()
    # CIQ_test_Wu()
    # CIQ_test_SFLA()
