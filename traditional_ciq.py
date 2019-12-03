import numpy as np
import cv2
from btpd import BTPD, Ueda_CIQ
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from proposal import CIQ_test, get_importance_error, ciq_eval_set
from img_util import mapping_pallet_to_img, compare_labmse, get_saliency_hist, get_saliency_upper_th, pil2cv, cv2pil,\
    get_allcolors_from_img
from skimage.measure import compare_nrmse, compare_psnr
import pandas as pd
import os
import time
from mediancut import median_cut


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


def SFLA_CQ(img, k, n_frogs=20, n_mem=5, T_max=100, J_max=5, rho=0.5):
    perm = np.arange(0, n_frogs, 1)
    hist_frogs = []
    hist_bestfrogs = []
    all_colors = get_allcolors_from_img(img)

    def eval_frog(frog):
        # 1. mapping
        mapped = mapping_pallet_to_img(img, frog)

        # 2. replace each element of frog
        new_frog = frog.copy()
        for n in range(k):
            indices = np.where(mapped == frog[n])
            org_pixels = img[indices[:2]]
            new_frog[n] = np.mean(org_pixels, axis=0)

        # 3. eval psnr
        psnr = compare_psnr(img, mapped)
        return psnr, new_frog

    def create_frog():
        indices = np.random.permutation(len(all_colors))
        return all_colors[indices[:k]]

    init_frogs = np.array([create_frog() for _ in range(n_frogs)])
    shuffled_frogs = init_frogs.copy()
    shuffled_fitness = np.zeros(shape=n_frogs)
    hist_frogs.append(shuffled_frogs)
    global_best_index = 0
    for t in range(T_max):
        shuffled_index = np.random.permutation(perm)
        shuffled_frogs = shuffled_frogs[shuffled_index]

        # eval all frogs
        for n in range(n_frogs):
            eval, new_frog = eval_frog(shuffled_frogs[n])
            shuffled_fitness[n] = eval
            shuffled_frogs[n] = new_frog

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
                next_x_worst_fit, next_x_worst = eval_frog(next_x_worst)

                if next_x_worst_fit < shuffled_fitness[mem_worst_index + index_bius]:
                    # apply Eqs.(1) and (3)
                    D = rho * (shuffled_frogs[global_best_index] - shuffled_frogs[mem_worst_index + bd])
                    next_x_worst = shuffled_frogs[mem_worst_index + bd] + D
                    next_x_worst_fit, next_x_worst = eval_frog(next_x_worst)

                    if next_x_worst_fit < shuffled_fitness[mem_worst_index + index_bius]:
                        # Move the worst frog to a random position
                        next_x_worst = create_frog()
                shuffled_frogs[mem_worst_index + index_bius] = next_x_worst
                shuffled_fitness[mem_worst_index + index_bius] = next_x_worst_fit

    return shuffled_frogs[global_best_index], hist_bestfrogs, hist_frogs


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


def BTPD_CIQ(img, M):
    """
    Binary Splittingによってカラーパレットを設計する．
    マッピング部も様々なアルゴリズムが提案されているため，カラーパレットとマッピング部で分割している．
    :param img: OpenCVによって得た入力画像
    :param M: 量子化数，カラーパレットサイズ
    :return:
    """
    S = np.reshape(img, (img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
    color_palette, root = BTPD(S, M)

    return color_palette, root


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


def CIQ_test_BTPD(M=[16], DIR=['sumple_img']):
    test_config = {
        'trans_flag': True,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': True,
        'save_tmpSM': True,
        'view_importance': False,
        'importance_eval': get_importance_error,
        'ciq_error_eval': ciq_eval_set(),
        'save_tmp_imgs': False
    }
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2LAB
            code_inverse = cv2.COLOR_LAB2BGR

            def ciq(img):
                trans_img = cv2.cvtColor(img, code)
                S = np.reshape(trans_img, newshape=(img.shape[0] * img.shape[1], 1, 3)).astype(np.uint64)
                all_colors = get_allcolors_from_img(img)
                q, root, groups = BTPD(S, m, visualization=True)
                reshape_q = np.reshape(q, newshape=(m, 1, 3)).astype(np.uint8)
                retrans_q = cv2.cvtColor(reshape_q, code_inverse)
                dict = {'palette': q,
                        'groups': [retrans_q[:, 0, :], all_colors]}
                return dict

            SAVE = 'BTPD_M{}_{}_LAB'.format(m, dir)
            CIQ_test(ciq, SAVE, test_img=dir, **test_config)


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


def CIQ_test_KMeans(M=[16], DIR=['sumple_img']):
    test_config = {
        'trans_flag': False,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': False,
        'save_tmpSM': True,
        'view_importance': True,
        'importance_eval': get_importance_error,
        'ciq_error_eval': ciq_eval_set(),
        'save_tmp_imgs': False
    }
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2Lab
            code_inverse = cv2.COLOR_Lab2BGR

            def ciq(img):
                # img = cv2.cvtColor(img, code)
                S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], img.shape[2]))
                kmeans, q = KMeans_CIQ(S, m)
                dict = {'palette': kmeans.cluster_centers_}
                return dict

            SAVE = 'KMeans_M{}_{}_RGB'.format(m, dir)
            CIQ_test(ciq, SAVE, dir, **test_config)


def CIQ_test_MedianCut(M=[16], DIR=['sumple_img']):
    test_config = {
        'trans_flag': False,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': True,
        'save_tmpSM': True,
        'view_importance': False,
        'importance_eval': get_importance_error,
        'ciq_error_eval': ciq_eval_set(),
        'save_tmp_imgs': False
    }
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2Lab
            code_inverse = cv2.COLOR_Lab2BGR

            def ciq(img):
                # img = cv2.cvtColor(img, code)
                pilimg = cv2pil(img)
                q = median_cut(pilimg, m)
                cv_q = q[:, ::-1]
                all_colors = get_allcolors_from_img(img)
                dict = {'palette': cv_q,
                        'groups': [cv_q, all_colors]}
                return dict

            SAVE = 'MedianCut_M{}_{}_RGB'.format(m, dir)
            CIQ_test(ciq, SAVE, dir, **test_config)


def CIQ_test_SFLA(M=[16], DIR=['sumple_img']):
    test_config = {
        'trans_flag': False,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': True,
        'save_tmpSM': False,
        'view_importance': True,
        'importance_eval': get_importance,
        'ciq_error_eval': ciq_eval_set(),
        'save_tmp_imgs': False
    }
    for dir in DIR:
        for m in M:
            def ciq(img):
                frog, _, __ = SFLA_CQ(img, m)
                dict = {'palette': frog,
                        'groups': [frog]}
                return dict

            SAVE = 'SFLA_M{}_{}'.format(m, dir)
            CIQ_test(ciq, SAVE, test_img=dir, **test_config)


def CIQ_test_Ueda(M=[16], DIR=['sumple_img']):
    test_config = {
        'trans_flag': False,
        'trans_code': cv2.COLOR_BGR2LAB,
        'trans_inverse_code': cv2.COLOR_LAB2BGR,
        'view_distribution': False,
        'save_tmpSM': True,
        'view_importance': True,
        'importance_eval': get_importance_error,
        'ciq_error_eval': ciq_eval_set(),
        'save_tmp_imgs': False
    }
    for dir in DIR:
        for m in M:
            code = cv2.COLOR_BGR2LAB
            inverse_code = cv2.COLOR_LAB2BGR

            def ciq(img):
                trans_img = cv2.cvtColor(img, code)
                S = np.reshape(img, newshape=(img.shape[0] * img.shape[1], img.shape[2]))
                _, __, Sv_map = get_saliency_hist(trans_img, sm='SR')
                Sv = np.reshape(Sv_map / 255.0, newshape=(len(S), 1, 1)).astype(np.float32)
                Sv += 1e-8
                q, groups = Ueda_CIQ(S, m, Sv)
                dict = {'palette': q,
                        'groups': groups}
                return dict

            SAVE = 'Ueda_M{}_{}_RGB'.format(m, dir)
            CIQ_test(ciq, SAVE, dir, **test_config)


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


if __name__ == '__main__':
    # CIQ_test_SFLA(M=[16], DIR=['sumple_img'])
    CIQ_test_Ueda(M=[16, 32], DIR=['sumple_img'])
    CIQ_test_BTPD(M=[16, 32], DIR=['sumple_img'])
    CIQ_test_MedianCut(M=[16, 32], DIR=['sumple_img'])
    CIQ_test_KMeans(M=[16, 32], DIR=['sumple_img'])
