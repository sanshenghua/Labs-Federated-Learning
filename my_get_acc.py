from plots_func.common_funcs import get_acc_loss
import matplotlib.pyplot as plt
import numpy as np
def loss_acc_evo(
    kept: list,
    dataset: str,
    sampling: str,
    n_SGD: int,
    seed: int,
    lr: float,
    decay: float,
    p: float,
    mu: float,
    n_rows: int,
    n_cols: int,
    axis,
    idx_row: int,
    plot_names: str,
):
    #weights = weights_clients(dataset)

    acc_hists, legend = get_acc_loss(
        dataset, sampling, "acc", n_SGD, seed, lr, decay, p, mu
    )

    print("============")
    #print(legend)
    #print(len(acc_hists))


def plot_fig_CIFAR10_alpha_effect_both(n_SGD: int, p: float, mu: float):#f4

    kept = ["MD", "Alg. 2", "Alg. 1"]
    plot_names = ["(a)", "(b)", "(c)"]

    n_rows, n_cols = 3, 2
    dataset_base = "CIFAR10"

    fig, axis = plt.subplots(n_rows, n_cols, figsize=(4.5, 6))

    # INFLUENCE OF THE NON-IID ASPECT
    sampling = "clustered_1"
    seed = 0
    decay = 1.0

    # INFLUENCE OF THE NUMBER OF SGD
    dataset = f"{dataset_base}_nbal_0.1"
    #print(dataset)
    loss_acc_evo(
        kept,
        f"{dataset_base}_nbal_0.1",
        sampling,
        n_SGD,
        seed,
        0.05,
        decay,
        p,
        mu,
        n_rows,
        n_cols,
        axis,
        0,
        plot_names,
    )

n_SGD = 100
p = 0.1
mu = 0.0
plot_fig_CIFAR10_alpha_effect_both(n_SGD, p, mu)