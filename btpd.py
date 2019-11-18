import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from pathlib import Path


class Node:
    def __init__(self, parent=None, data=None, height=None):
        self.parent = parent
        self.data = data
        self.left = None
        self.right = None
        self.height = height

    def set_right(self, right):
        self.right = right

    def set_left(self, left):
        self.left = left

    def set_data(self, data):
        for key in data.keys():
            self.data[key] = data[key]

    def set_height(self):
        height = 0
        node = self
        while node.get_parent() is not None:
            height += 1
            node = node.get_parent()
            if node.get_height() is not None:
                height += node.get_height()
                break
        self.height = height

    def get_data(self):
        return self.data

    def get_right(self):
        return self.right

    def get_left(self):
        return self.left

    def get_parent(self):
        return self.parent

    def get_allparams(self):
        parent = self.parent
        right = self.right
        left = self.left
        data = self.data
        return [parent, right, left, data]

    def get_height(self):
        return self.height

    def preorder(self, func):
        func(self)
        left = self.get_left()
        right = self.get_right()
        if left is not None:
            left.preorder(func)

        if right is not None:
            right.preorder(func)


class RootNode(Node):
    def __init__(self, parent=None, data=None):
        super().__init__(parent=parent, data=data, height=0)
        self.leaves = []
        self.root = self

    def set_leaves(self):
        self.leaves = []

        def check_leave(node):
            if node.get_left() is None and node.get_right() is None:
                node.root.set_leave(node)
        self.preorder(check_leave)
        return self.leaves

    def set_leave(self, node):
        self.leaves.append(node)

    def get_leaves(self):
        return self.leaves

    def get_bottomleft(self):
        bottom_left = self.get_left()
        while bottom_left.get_left() is not None:
            bottom_left = bottom_left.get_left()

        return bottom_left


class SubNode(Node):
    def __init__(self, root, parent=None, data=None, height=None):
        super().__init__(parent=parent, data=data, height=height)
        self.root = root


def get_R(c, Sv=None):
    # 要するに，分散共分散行列を算出するためのものE(XiXj)だと思われるので，
    # 論文のまま記述すると，総数で割られてないのでおかしい
    sum = np.zeros(shape=(3, 3))
    if Sv is None:
        Sv = np.ones(len(c))

    for s, sv in zip(c, Sv):
        tmp = sv * s * s.T
        sum += tmp
    return sum


def get_m(c):
    sum = np.sum(c, axis=0)
    # sum = c[0].copy()
    # for s in c[1:]:
    #     sum += s
    return sum


def get_N(c):
    return len(c)


def get_params(S, r, index=None):
    m = get_m(S)
    N = get_N(S)
    R = np.sum(r, axis=0)
    q = m / N

    tmp = (m * m.T) / N
    R_ = R - tmp
    W, v = np.linalg.eig(R_)
    ev = np.max(W)
    # var = np.var(S)
    e = v[np.argmax(W)]

    return {'S': S, 'm': m, 'N': N, 'R': R, 'q': q, 'e': e, 'max_ev': ev, 'r': r, 'index': index}


def get_params_with_weight(S, Sv, weighted_r, weighted_m, index=None):
    m = np.sum(weighted_m, axis=0)
    N = np.sum(Sv)
    R = np.sum(weighted_r, axis=0)
    q = m / N

    tmp = (m * m.T) / N
    R_ = R - tmp
    W, v = np.linalg.eig(R_)
    ev = np.max(W)
    e = v[np.argmax(W)]

    return {'S': S, 'm': m, 'N': N, 'R': R, 'q': q, 'e': e, 'max_ev': ev, 'Sv': Sv,
            'weighted_r': weighted_r, 'weighted_m': weighted_m, 'index': index}


def get_params_for_bst(S1, S2, r1, r2, parent_params, index1=None, index2=None):
    right_params = get_params(S1, r1, index=index1)
    m = parent_params['m'] - right_params['m']
    N = parent_params['N'] - right_params['N']
    R = parent_params['R'] - right_params['R']
    q = m / N
    tmp = (m * m.T) / N
    R_ = R - tmp
    W, v = np.linalg.eig(R_)
    ev = np.max(W)
    e = v[np.argmax(W)]
    left_params = {'S': S2, 'm': m, 'N': N, 'R': R, 'q': q, 'e': e, 'max_ev': ev, 'r': r2, 'index': index2}
    # left_params = get_params(S2)
    return right_params, left_params


def get_params_for_bst_with_weight(S1, S2, Sv1, weighted_r1, weighted_m1,
                                   Sv2, weighted_r2, weighted_m2, index1=None, index2=None):
    right_params = get_params_with_weight(S1, Sv1, weighted_r1, weighted_m1, index=index1)
    # m = parent_params['m'] - right_params['m']
    # N = parent_params['N'] - right_params['N']
    # R = parent_params['R'] - right_params['R']
    # q = m / N
    # tmp = (m * m.T) / N
    # R_ = R - tmp
    # W, v = np.linalg.eig(R_)
    # ev = np.max(W)
    # e = v[np.argmax(W)]
    # left_params = {'S': S2, 'm': m, 'N': N, 'R': R, 'q': q, 'e': e, 'max_ev': ev, 'Sv': Sv2}
    left_params = get_params_with_weight(S2, Sv2, weighted_r2, weighted_m2, index=index2)
    return right_params, left_params


def BTPD(S, M, **visualize_config):
    visualization = visualize_config['visualization']
    if visualization:
        visualize_path = visualize_config['visualize_path']
    # precalc
    n_ch = S.shape[-1]
    S = np.reshape(S, newshape=(len(S), 1, n_ch)).astype(np.uint64)
    pre_r = np.array([s * s.T for s in S])
    pre_r = np.reshape(pre_r, newshape=(len(S), n_ch, n_ch))

    params = get_params(S, pre_r, index=np.array([n for n in range(len(S))]))
    root = RootNode(parent=None, data=params)
    palette = []
    for num in range(M - 1):
        leaves = root.set_leaves()
        max_ev_arr = np.array([leaf.get_data()['max_ev'] for leaf in leaves])
        current_node = leaves[int(np.argmax(max_ev_arr))]

        data = current_node.get_data()
        current_S = data['S']
        current_r = data['r']
        current_q = data['q']
        current_e = data['e']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        n_index_in_S = data['index'][c_2n_index]
        n1_index_in_S = data['index'][c_2n1_index]
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, n_ch))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, n_ch))

        r_2n = np.reshape(current_r[c_2n_index[0]], (num_c2n, n_ch, n_ch))
        r_2n1 = np.reshape(current_r[c_2n1_index[0]], (num_c2n1, n_ch, n_ch))

        left_params, right_params = get_params_for_bst(c_2n, c_2n1, r_2n, r_2n1, current_node.get_data(),
                                                       index1=n_index_in_S, index2=n1_index_in_S)
        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)

        if visualization:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)  # 何もプロットしていないAxesでもAxisは自動的に作られる
            colors_1 = c_2n[:, 0, :] / 255.0
            colors_2 = c_2n1[:, 0, :] / 255.0
            tmp_y1 = np.array([0 for n in range(len(c_2n_index[0]))])
            tmp_y2 = np.array([0 for n in range(len(c_2n1_index[0]))])
            ax.scatter(compare[c_2n_index], tmp_y1, c=colors_1, marker=',', label='pixels')
            ax.scatter(compare[c_2n1_index], tmp_y2, c=colors_2, marker=',', label='pixels')

            cri_y = np.arange(-0.1, 0.1, 0.001)
            criterias = [criteria for n in range(len(cri_y))]
            ax.plot(criterias, cri_y, c='blue', linestyle='dashed',)
            ax.set_xlabel("principal score")
            # ax.set_ylabel("G")
            # ax.set_xlim([0, 255])
            # ax.set_ylim([-1, 1])
            ax.tick_params(labelleft="off", left="off")  # y軸の削除
            ax.set_yticklabels([])
            ax.grid()
            Path(visualize_path).mkdir(parents=True, exist_ok=True)
            filepath = os.path.join(visualize_path, f'{num}.jpg')
            fig.savefig(filepath, bbox_inches="tight", transparent=True)
            plt.close(fig)

    leaves = root.set_leaves()
    groups = []
    for leaf in leaves:
        params = leaf.get_data()
        groups.append(np.reshape(params['S'], newshape=(len(params['S']), 3)))
        palette.append(params['q'])

    palette = np.array(palette)
    color_palette = np.round(palette)
    return color_palette, root, np.array(groups)


def BTPD_LimitationSv(S, limit):
    # precalc
    n_ch = S.shape[-1]
    S = np.reshape(S, newshape=(len(S), 1, n_ch)).astype(np.uint64)
    pre_r = np.array([s * s.T for s in S])
    pre_r = np.reshape(pre_r, newshape=(len(S), n_ch, n_ch))

    params = get_params(S, pre_r, index=np.array([n for n in range(len(S))]))
    root = RootNode(parent=None, data=params)
    palette = []
    max_ev = np.inf
    num = 0
    while limit < max_ev:
        leaves = root.set_leaves()
        max_ev_arr = np.array([leaf.get_data()['max_ev'] for leaf in leaves])
        max_ev = np.max(max_ev_arr)
        current_node = leaves[int(np.argmax(max_ev_arr))]

        data = current_node.get_data()
        current_S = data['S']
        current_r = data['r']
        current_q = data['q']
        current_e = data['e']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        n_index_in_S = data['index'][c_2n_index]
        n1_index_in_S = data['index'][c_2n1_index]
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, n_ch))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, n_ch))

        r_2n = np.reshape(current_r[c_2n_index[0]], (num_c2n, n_ch, n_ch))
        r_2n1 = np.reshape(current_r[c_2n1_index[0]], (num_c2n1, n_ch, n_ch))

        left_params, right_params = get_params_for_bst(c_2n, c_2n1, r_2n, r_2n1, current_node.get_data(),
                                                       index1=n_index_in_S, index2=n1_index_in_S)
        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)
        num += 1

    leaves = root.set_leaves()
    groups = []
    for leaf in leaves:
        params = leaf.get_data()
        groups.append(np.reshape(params['S'], newshape=(len(params['S']), 3)))
        palette.append(params['q'])

    palette = np.array(palette)
    color_palette = np.round(palette)
    return color_palette, root, np.array(groups)


def BTPD_WTSE(S, M, Sv):
    # precalc
    S = np.reshape(S, newshape=(len(S), 1, 3)).astype(np.uint32)
    Sv = np.reshape(Sv, newshape=(len(S), 1, 1)).astype(np.float32)
    pre_m = np.array([w * s for s, w in zip(S, Sv)])
    pre_R = np.array([m * s.T for m, s in zip(pre_m, S)])
    pre_m = np.reshape(pre_m, newshape=(len(S), 1, 3))
    pre_R = np.reshape(pre_R, newshape=(len(S), 3, 3))

    params = get_params_with_weight(S, Sv, pre_R, pre_m, index=np.array([n for n in range(len(S))]))
    root = RootNode(parent=None, data=params)
    palette = []
    for num in range(M - 1):
        leaves = root.set_leaves()
        max_ev = leaves[0].get_data()['max_ev']
        current_node = leaves[0]
        for leaf in leaves[1:]:
            params = leaf.get_data()
            ev = params['max_ev']
            if max_ev < ev:
                current_node = leaf
                max_ev = ev

        data = current_node.get_data()
        current_S = data['S']
        current_Sv = data['Sv']
        current_wr = data['weighted_r']
        current_wm = data['weighted_m']
        current_q = data['q']
        current_e = data['e']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, 3))
        sv_2n = np.reshape(current_Sv[c_2n_index[0]], (num_c2n, 1))
        sv_2n1 = np.reshape(current_Sv[c_2n1_index[0]], (num_c2n1, 1))
        weighted_r_2n = np.reshape(current_wr[c_2n_index[0]], (num_c2n, 3, 3))
        weighted_m_2n = np.reshape(current_wm[c_2n_index[0]], (num_c2n, 1, 3))
        weighted_r_2n1 = np.reshape(current_wr[c_2n1_index[0]], (num_c2n1, 3, 3))
        weighted_m_2n1 = np.reshape(current_wm[c_2n1_index[0]], (num_c2n1, 1, 3))
        n_index_in_S = data['index'][c_2n_index]
        n1_index_in_S = data['index'][c_2n1_index]

        left_params, right_params = get_params_for_bst_with_weight(c_2n, c_2n1, sv_2n, weighted_r_2n, weighted_m_2n,
                                                                   sv_2n1, weighted_r_2n1, weighted_m_2n1,
                                                                   n_index_in_S, n1_index_in_S)
        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)

    leaves = root.set_leaves()
    groups = []
    for leaf in leaves:
        params = leaf.get_data()
        groups.append(np.reshape(params['S'], newshape=(len(params['S']), 3)))
        palette.append(params['q'])

    palette = np.array(palette)
    color_palette = np.round(palette)
    return color_palette, root, np.array(groups)


def BTPD_WTSE_LimitationSv(S, Sv, limit):
    # precalc
    S = np.reshape(S, newshape=(len(S), 1, 3)).astype(np.uint32)
    Sv = np.reshape(Sv, newshape=(len(S), 1, 1)).astype(np.float32)
    pre_m = np.array([w * s for s, w in zip(S, Sv)])
    pre_R = np.array([m * s.T for m, s in zip(pre_m, S)])
    pre_m = np.reshape(pre_m, newshape=(len(S), 1, 3))
    pre_R = np.reshape(pre_R, newshape=(len(S), 3, 3))

    params = get_params_with_weight(S, Sv, pre_R, pre_m, index=np.array([n for n in range(len(S))]))
    root = RootNode(parent=None, data=params)
    palette = []
    max_ev = np.inf
    num = 0
    while limit < max_ev:
        leaves = root.set_leaves()
        max_ev_arr = np.array([leaf.get_data()['max_ev'] for leaf in leaves])
        max_ev = np.max(max_ev_arr)
        current_node = leaves[int(np.argmax(max_ev_arr))]

        data = current_node.get_data()
        current_S = data['S']
        current_Sv = data['Sv']
        current_wr = data['weighted_r']
        current_wm = data['weighted_m']
        current_q = data['q']
        current_e = data['e']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, 3))
        sv_2n = np.reshape(current_Sv[c_2n_index[0]], (num_c2n, 1))
        sv_2n1 = np.reshape(current_Sv[c_2n1_index[0]], (num_c2n1, 1))
        weighted_r_2n = np.reshape(current_wr[c_2n_index[0]], (num_c2n, 3, 3))
        weighted_m_2n = np.reshape(current_wm[c_2n_index[0]], (num_c2n, 1, 3))
        weighted_r_2n1 = np.reshape(current_wr[c_2n1_index[0]], (num_c2n1, 3, 3))
        weighted_m_2n1 = np.reshape(current_wm[c_2n1_index[0]], (num_c2n1, 1, 3))
        n_index_in_S = data['index'][c_2n_index]
        n1_index_in_S = data['index'][c_2n1_index]

        left_params, right_params = get_params_for_bst_with_weight(c_2n, c_2n1, sv_2n, weighted_r_2n, weighted_m_2n,
                                                                   sv_2n1, weighted_r_2n1, weighted_m_2n1,
                                                                   n_index_in_S, n1_index_in_S)
        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)

    leaves = root.set_leaves()
    groups = []
    for leaf in leaves:
        params = leaf.get_data()
        groups.append(np.reshape(params['S'], newshape=(len(params['S']), 3)))
        palette.append(params['q'])

    palette = np.array(palette)
    color_palette = np.round(palette)
    return color_palette, root, np.array(groups)


def BTPD_PaletteDeterminationFromSV(S, M, Sv):
    """
    顕著性マップのみで二分木を分割していく
    --> 顕著度の高い色が保存されるかと思ったけどダメ
    :param S:
    :param M:
    :param Sv:
    :return:
    """
    # precalc
    S = np.reshape(S, newshape=(len(S), 1, 3)).astype(np.uint64)
    pre_r = np.array([s * s.T for s in S])
    pre_r = np.reshape(pre_r, newshape=(len(S), 3, 3))
    Sv = np.reshape(Sv, newshape=(len(S), 1, 1)).astype(np.float32)

    params = get_params(S, pre_r)
    params['Sv'] = Sv
    root = RootNode(parent=None, data=params)
    palette = []
    for num in range(M - 1):
        leaves = root.set_leaves()
        max_ev = leaves[0].get_data()['max_ev']
        current_node = leaves[0]
        for leaf in leaves[1:]:
            params = leaf.get_data()
            ev = params['max_ev']
            if max_ev < ev:
                current_node = leaf
                max_ev = ev

        data = current_node.get_data()
        current_S = data['S']
        current_Sv = data['Sv']
        current_r = data['r']
        current_q = data['q']
        current_e = data['e']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, 3))
        r_2n = np.reshape(current_r[c_2n_index[0]], (num_c2n, 3, 3))
        r_2n1 = np.reshape(current_r[c_2n1_index[0]], (num_c2n1, 3, 3))
        sv_2n = np.reshape(current_Sv[c_2n_index[0]], (num_c2n, 1))
        sv_2n1 = np.reshape(current_Sv[c_2n1_index[0]], (num_c2n1, 1))

        left_params, right_params = get_params_for_bst(c_2n, c_2n1, r_2n, r_2n1, current_node.get_data())
        left_params['Sv'] = sv_2n
        right_params['Sv'] = sv_2n1
        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)

    leaves = root.set_leaves()
    for leaf in leaves:
        params = leaf.get_data()
        palette.append(SvweightedPaletteDetermination(params['S'], params['Sv']))

    palette = np.array(palette)
    color_palette = np.round(palette)
    return color_palette


def BTPD_InitializationFromSv(S, M, Sv, weights):
    """
    最初数回を，顕著性マップのみで分割する
    その後，顕著度に基づく重みを用いて色空間上を分割する．
    --> 顕著性マップのみ，っていうのがダメっぽい
    :param S:
    :param M:
    :param Sv:
    :param weights:
    :return:
    """
    # precalc
    S = np.reshape(S, newshape=(len(S), 1, 3)).astype(np.uint32)
    W = np.reshape(weights, newshape=(len(S), 1, 1)).astype(np.float32)

    palette = []
    M0 = 2
    __, root = BTPD(Sv, M0)
    #
    leaves = root.set_leaves()
    for leave in leaves:
        data = leave.get_data()
        index = data['index']
        current_S = S[index]
        current_W = W[index]

        pre_m = np.array([w * s for s, w in zip(current_S, current_W)])
        pre_R = np.array([m * s.T for m, s in zip(pre_m, current_S)])
        pre_m = np.reshape(pre_m, newshape=(len(current_S), 1, 3))
        pre_R = np.reshape(pre_R, newshape=(len(current_S), 3, 3))
        params = get_params_with_weight(current_S, current_W, pre_R, pre_m)
        leave.set_data(params)

    for num in range(M - M0):
        leaves = root.set_leaves()
        max_ev = leaves[0].get_data()['max_ev']
        current_node = leaves[0]
        for leaf in leaves[1:]:
            params = leaf.get_data()
            ev = params['max_ev']
            if max_ev < ev:
                current_node = leaf
                max_ev = ev

        data = current_node.get_data()
        current_S = data['S']
        current_Sv = data['Sv']
        current_wr = data['weighted_r']
        current_wm = data['weighted_m']
        current_q = data['q']
        current_e = data['e']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, 3))
        sv_2n = np.reshape(current_Sv[c_2n_index[0]], (num_c2n, 1))
        sv_2n1 = np.reshape(current_Sv[c_2n1_index[0]], (num_c2n1, 1))
        weighted_r_2n = np.reshape(current_wr[c_2n_index[0]], (num_c2n, 3, 3))
        weighted_m_2n = np.reshape(current_wm[c_2n_index[0]], (num_c2n, 1, 3))
        weighted_r_2n1 = np.reshape(current_wr[c_2n1_index[0]], (num_c2n1, 3, 3))
        weighted_m_2n1 = np.reshape(current_wm[c_2n1_index[0]], (num_c2n1, 1, 3))

        left_params, right_params = get_params_for_bst_with_weight(c_2n, c_2n1, sv_2n, weighted_r_2n, weighted_m_2n,
                                                                   sv_2n1, weighted_r_2n1, weighted_m_2n1,
                                                                   current_node.get_data())
        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)

    leaves = root.set_leaves()
    for leaf in leaves:
        params = leaf.get_data()
        palette.append(params['q'])

    palette = np.array(palette)
    color_palette = np.round(palette)
    return color_palette, root


def BTPD_InitializationFromIncludingSv(S, M, Sv, weights):
    """
    最初数回を，顕著性マップ + 色空間で分割する
    その後，顕著度に基づく重みを用いて色空間上を分割する．
    --> 顕著性マップのみ，っていうのがダメっぽい
    :param S:
    :param M:
    :param Sv:
    :param weights:
    :return:
    """
    # precalc
    S = np.reshape(S, newshape=(len(S), 1, 3)).astype(np.uint32)
    W = np.reshape(weights, newshape=(len(S), 1, 1)).astype(np.float32)
    S_Sv = np.concatenate([S, Sv], axis=2)

    palette = []
    M0 = int(M / 4)
    __, root = BTPD(S_Sv, M0)
    #
    leaves = root.set_leaves()
    for leave in leaves:
        data = leave.get_data()
        index = data['index']
        current_S = S[index]
        current_W = W[index]

        pre_m = np.array([w * s for s, w in zip(current_S, current_W)])
        pre_R = np.array([m * s.T for m, s in zip(pre_m, current_S)])
        pre_m = np.reshape(pre_m, newshape=(len(current_S), 1, 3))
        pre_R = np.reshape(pre_R, newshape=(len(current_S), 3, 3))
        params = get_params_with_weight(current_S, current_W, pre_R, pre_m, index=index)
        leave.set_data(params)

    for num in range(M - M0):
        leaves = root.set_leaves()
        max_ev = leaves[0].get_data()['max_ev']
        current_node = leaves[0]
        for leaf in leaves[1:]:
            params = leaf.get_data()
            ev = params['max_ev']
            if max_ev < ev:
                current_node = leaf
                max_ev = ev

        data = current_node.get_data()
        current_S = data['S']
        current_Sv = data['Sv']
        current_wr = data['weighted_r']
        current_wm = data['weighted_m']
        current_q = data['q']
        current_e = data['e']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        n_index_in_S = data['index'][c_2n_index]
        n1_index_in_S = data['index'][c_2n1_index]
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, 3))
        sv_2n = np.reshape(current_Sv[c_2n_index[0]], (num_c2n, 1))
        sv_2n1 = np.reshape(current_Sv[c_2n1_index[0]], (num_c2n1, 1))
        weighted_r_2n = np.reshape(current_wr[c_2n_index[0]], (num_c2n, 3, 3))
        weighted_m_2n = np.reshape(current_wm[c_2n_index[0]], (num_c2n, 1, 3))
        weighted_r_2n1 = np.reshape(current_wr[c_2n1_index[0]], (num_c2n1, 3, 3))
        weighted_m_2n1 = np.reshape(current_wm[c_2n1_index[0]], (num_c2n1, 1, 3))

        left_params, right_params = get_params_for_bst_with_weight(c_2n, c_2n1, sv_2n, weighted_r_2n, weighted_m_2n,
                                                                   sv_2n1, weighted_r_2n1, weighted_m_2n1,
                                                                   index1=n_index_in_S, index2=n1_index_in_S)
        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)

    leaves = root.set_leaves()
    for leaf in leaves:
        params = leaf.get_data()
        palette.append(params['q'])

    palette = np.array(palette)
    color_palette = np.round(palette)
    return color_palette, root


def SvweightedPaletteDetermination(pixels, Sv):
    return (np.sum(pixels[:, 0, :] * Sv, axis=0) / np.sum(Sv)).astype(np.uint8)


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


def SMBW_BTPD(S, Sv, M, M0=0.8, R=2):
    M0 = int(M0 * M)
    if len(S.shape) != 3:
        S = np.reshape(S, newshape=(len(S), 1, 3)).astype(np.uint64)

    if len(Sv.shape) != 2:
        Sv = np.reshape(Sv, newshape=(len(S), 1))

    params = get_params(S)
    params['sv'] = Sv
    root = RootNode(parent=None, data=params)
    palette = []
    for num in range(M0 - 1):
        leaves = root.set_leaves()
        max_ev = leaves[0].get_data()['max_ev']
        current_node = leaves[0]
        for leaf in leaves[1:]:
            params = leaf.get_data()
            ev = params['max_ev']
            if max_ev < ev:
                current_node = leaf
                max_ev = ev
        data = current_node.get_data()
        current_S = data['S']
        current_q = data['q']
        current_e = data['e']
        current_Sv = data['sv']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, 3))
        sv_2n = current_Sv[c_2n_index]
        sv_2n1 = current_Sv[c_2n1_index]

        left_params, right_params = get_params_for_bst(c_2n, c_2n1, current_node.get_data())
        left_params['sv'] = sv_2n
        right_params['sv'] = sv_2n1

        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)

    # 重みを考慮する
    for num in range(M0 - 1, M - 1):
        leaves = root.set_leaves()
        params = leaves[0].get_data()
        ev = params['max_ev']
        sv = np.sum(params['sv'] ** 2)
        # print('sv: {} \t ev: {}'.format(sv, ev))
        # sv = np.sum(params['sv']) * R
        max_wev = sv * ev
        current_node = leaves[0]
        for leaf in leaves[1:]:
            params = leaf.get_data()
            ev = params['max_ev']
            sv = np.sum(params['sv'] ** 2)
            # print('sv: {} \t ev: {}'.format(sv, ev))
            # sv = np.sum(params['sv']) * R
            wev = sv * ev
            if max_wev < wev:
                current_node = leaf
                max_wev = wev

        data = current_node.get_data()
        current_S = data['S']
        current_q = data['q']
        current_e = data['e']
        current_Sv = data['sv']
        criteria = np.dot(current_e, current_q[0])
        compare = np.dot(current_e, current_S[:, 0, :].T)
        c_2n_index = np.where(compare <= criteria)
        c_2n1_index = np.where(compare > criteria)
        num_c2n = len(c_2n_index[0])
        num_c2n1 = len(c_2n1_index[0])

        if num_c2n1 <= 0:
            # 分割できない
            # 分散が相当低いはずなので，本来選ばれるはずのない状態
            print('could not separate the extraction')
            break

        # 現ノードから子の作成
        c_2n = np.reshape(current_S[c_2n_index[0]], (num_c2n, 1, 3))
        c_2n1 = np.reshape(current_S[c_2n1_index[0]], (num_c2n1, 1, 3))
        sv_2n = current_Sv[c_2n_index]
        sv_2n1 = current_Sv[c_2n1_index]

        left_params, right_params = get_params_for_bst(c_2n, c_2n1, current_node.get_data())
        left_params['sv'] = sv_2n
        right_params['sv'] = sv_2n1

        right = SubNode(parent=current_node, data=right_params, height=num + 1, root=root)
        left = SubNode(parent=current_node, data=left_params, height=num + 1, root=root)
        current_node.set_right(right=right)
        current_node.set_left(left=left)

    leaves = root.set_leaves()
    for leaf in leaves:
        params = leaf.get_data()
        palette.append(params['q'])

    palette = np.array(palette)
    color_palette = np.round(palette)
    return color_palette


def Weighted_PCA(X, W):
    A = np.zeros(shape=(3, 3))
    sum_w = np.sum(W)
    X_ = np.sum(np.multiply(W, X), axis=0) / sum_w
    W = np.reshape(W, newshape=(len(W)))
    for i in range(3):
        for j in range(3):
            tmp = np.multiply((X[:, i] - X_[i]), (X[:, j] - X_[j])).astype(np.float32)
            A[i, j] = np.dot(W, tmp) / sum_w

    W, v = np.linalg.eig(A)
    ev = np.max(W)
    e = v[np.argmax(W)]
    l = ev * sum_w
    dl = np.dot((X - X_), e)
    return l, dl


def Ueda_CIQ(S, M, Sv):
    # precalc
    S = np.reshape(S, newshape=(len(S), 3)).astype(np.int32)
    Sv = np.reshape(Sv, newshape=(len(S), 1)).astype(np.float32)

    # クラス番号の割り当て
    class_labels = np.zeros(shape=len(S))
    for n in range(M - 1):
        # 分割対象クラスの決定
        current_l = 0
        current_index = np.where(class_labels == current_l)[0]
        current_S = S[current_index]
        current_Sv = Sv[current_index]
        max_val, dl = Weighted_PCA(current_S, current_Sv)
        if n > 0:
            for l in range(1, n + 1):
                index = np.where(class_labels == l)[0]
                val, tmp_dl = Weighted_PCA(S[index], Sv[index])
                if max_val < val:
                    current_index = index
                    current_Sv = Sv[index]
                    current_l = l
                    max_val = val
                    dl = tmp_dl

        # 大津の線形判別
        current_Sv = current_Sv / np.sum(current_Sv)
        sorted_dl_index = np.argsort(dl)
        sorted_index = current_index[sorted_dl_index]
        N = len(dl)
        sorted_dl = np.reshape(dl[sorted_dl_index], newshape=N).astype(np.float32)
        sorted_w = np.reshape(current_Sv[sorted_dl_index], newshape=N).astype(np.float32)

        cum1 = np.cumsum(sorted_w)[:N - 1]         # cum1[i]: i番目までの顕著度の総和
        cum2 = np.cumsum(sorted_w[::-1])[:N - 1][::-1]   # cum2[i] i番目以降の顕著度の総和
        w1_w2 = np.multiply(cum1, cum2)
        tmp = np.multiply(sorted_dl, sorted_w)
        tmp = np.cumsum(tmp)[:N - 1]
        m1 = tmp / cum1
        m2 = tmp / cum2

        tmp2 = (m1 - m2)
        g = np.multiply(w1_w2, np.power(tmp2, 2))
        d = np.argmax(g)

        if d == 0:
            q = []
            for n in range(M):
                index = np.where(class_labels == n)
                index_S = S[index]
                represent = np.mean(index_S, axis=0).astype(np.int)
                q.append(represent)
            print('d == 0:  {} colors CIQ'.format(n))
            return np.array(q)

        sorted_index1 = sorted_index[:d]
        sorted_index2 = sorted_index[d:]

        class_labels[sorted_index1] = current_l
        class_labels[sorted_index2] = n + 1

    q = []
    groups = []
    for n in range(M):
        index = np.where(class_labels == n)
        index_S = S[index]
        groups.append(index_S)
        represent = np.mean(index_S, axis=0).astype(np.int)
        q.append(represent)
    return np.array(q), groups

