import numpy as np


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


if __name__ == '__main__':
    OneMaxBySFLA()
