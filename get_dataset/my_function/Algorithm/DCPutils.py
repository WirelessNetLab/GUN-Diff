import numpy as np

def get_wireless_fronthaul_noise(P_max, channel, device_num, tau_c, tau_d, B_A, B_F, ue_num):
    """

    """
    compress_noise = {}
    for i in range(len(channel)):
        channel_cpu_uav = channel[f'uav_{i+1}']

        N0 = 4e-21
        N_0 = N0/2 * B_F * 1e+6

        channel_capacity = get_capacity(channel=channel_cpu_uav, Pmax=P_max[i], N0=N_0)/device_num # 归一化带宽下均分到的容量
        compress_noise_down = 2 ** (tau_c * channel_capacity * B_F/(tau_d * B_A * ue_num)) - 1
        compress_noise_temp = 1. / compress_noise_down
        compress_noise[f'uav_{i+1}'] = compress_noise_temp
    # print('Get Wireless Fronthaul Link Capacity Finished\n')
    return compress_noise

def get_capacity(args, channel, Pmax, N0=1e-20):

    channel_cpu_uav = channel
    U, S_diag, Vh = np.linalg.svd(channel_cpu_uav)
    p_opt = water_filling(S_diag, Pmax, N0)
    capacity = np.sum(np.log2(1 + (S_diag * p_opt) / N0))
    return capacity

def water_filling(s, P_total, sigma2, max_iter=1000, tol=1e-6):
    s_sq = np.abs(s)
    eps = 1e-12

    N = len(s_sq)

    mu_low = 0.0
    mu_high = P_total + np.max(sigma2 / s_sq)

    for _ in range(max_iter):
        mu = 0.5 * (mu_low + mu_high)
        p = np.maximum(mu - sigma2 / s_sq, 0.0)
        totau_power = np.sum(p)

        if abs(totau_power - P_total) / P_total < tol:
            break
        elif totau_power > P_total:
            mu_high = mu
        else:
            mu_low = mu

    return p


