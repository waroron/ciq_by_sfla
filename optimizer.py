import numpy as np
import os
import cv2
import time
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error


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
    for t in range(T_max):
        shuffled_index = np.random.permutation(perm)
        shuffled_frogs = shuffled_frogs[shuffled_index]
        shuffled_fitness = eval_frogs(shuffled_frogs)
        global_best_index = np.argmax(shuffled_fitness)
        hist_bestfrogs.append(shuffled_frogs[global_best_index])

        print('best frog: {} \n best score: {}'.format(shuffled_frogs[global_best_index],
                                                       shuffled_fitness[global_best_index]))

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


def CIQ_test_BTPD():
    DIR = 'sumple_img'
    M = 16
    imgs = os.listdir(DIR)

    for img_path in imgs:
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)
        st = time.time()
        q = BTPD_CIQ(img, M)
        en = time.time()
        print('{}: {} colors pallete by BTPD: time {}'.format(img_path, M, en - st))
        for pix in q:
            print(pix)

        print('\n\n')


def CIQ_test_PSO():
    DIR = 'sumple_img'
    K = 16
    t_max = 100
    n_p = 50
    p_kmeans = 0.5
    kmeans_iteration = 10
    w = 0.729
    c1 = 1.4955
    c2 = 1.4955
    imgs = os.listdir(DIR)

    for img_path in imgs:
        path = os.path.join(DIR, img_path)
        img = cv2.imread(path)
        st = time.time()
        q = PSO_CIQ(img, K, n_p, t_max, p_kmeans, kmeans_iteration, w, c1, c2)
        en = time.time()
        print('{}: {} colors pallete by PSO: time {}'.format(img_path, K, en - st))
        for pix in q:
            print(pix)

        print('\n\n')


if __name__ == '__main__':
    # CIQ_test_BTPD()
    CIQ_test_PSO()
    # OneMaxBySFLA()
