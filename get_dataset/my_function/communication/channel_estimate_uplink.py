import numpy as np
import time
def channel_estimate(args, channel, pilots, pilot_index, beta, p_k, tau_p, kesi_qk, test):
    """
    Args:
        channel     : channel |0| -> cpu & ap & ue |1|-> uav & ue & ue |2|
        pilots      : pilot, size = (pilot_lenth, pilot_lenth)
        pilot_index : pilot index for each UE size = ue_num
        beta        : Large Scale Fading, beta |0| -> beta_uav_ue & beta_ap_ue |1|
        p_k         : Normalized pilot power
        tau_p       : Pilot Length

    Returns:
        tuple:
            - channel_estimate_uav_ue : estimated channel uav -> ue 
            - channel_estimate_ap_ue  : estimated channel ap -> ue  

        
    """

    channel_uav_ue, channel_ap_ue  = channel['uav_ue'], channel['ap_ue']
    beta_uav_ue,    beta_ap_ue     = beta   ['uav_ue'], beta   ['ap_ue']
    Y_uavs = {}
    channel_estimate_uav_ue = {}
    sigma = args.B_A * args.N0/2 *1e6
    c_ap, c_uav = np.zeros((args.ap_num, args.ue_num)), np.zeros((args.uav_num, args.ue_num))
    gamma_ap, gamma_uav = np.zeros((args.ap_num, args.ue_num)), np.zeros((args.uav_num, args.ue_num))
    for i in range(len(channel_uav_ue)):

        uav_receive_pilot = np.zeros((channel_uav_ue[f'uav_{i+1}']['ue_1'].shape[1], pilots.shape[1]), dtype=np.complex128)# np.zeros((len(channel_uav_ue[f'uav_{i+1}']), pilots.shape[1])) # (uav_i+1 服务的ue数目, pilot_lenth)
        for j in range(len(channel_uav_ue[f'uav_{i+1}'])):
            uav_receive_pilot += np.sqrt(p_k * tau_p) * (channel_uav_ue[f'uav_{i+1}'][f'ue_{j+1}'].conj().T  @  pilots [pilot_index[j]][np.newaxis,:]) # 第j个用户的导频序列
        noise = complex_gaussian(sigma, size=(uav_receive_pilot.shape[0], pilots.shape[1]), test=test) # np.random.normal(0,10,size=(1, pilots.shape[1]))
        Y_uavs[f'uav_{i+1}'] = uav_receive_pilot + noise
    
    for i in range(len(channel_uav_ue)):
        channel_estimate_uav = {}
        for j in range(len(channel_uav_ue[f'uav_{i+1}'])):
            up = np.sqrt(p_k * tau_p) * beta_uav_ue[i,j]
            down = p_k * tau_p * (np.sum(beta_uav_ue[i, pilot_index == pilot_index[j]])) + sigma
            c_qk = up/down
            channel_estimate_uav[f'ue_{j+1}'] = c_qk * (Y_uavs[f'uav_{i+1}'] @ pilots[pilot_index[j]][np.newaxis,:].conj().T).conj().T
            c_uav[i,j] = c_qk
            gamma_uav[i,j] = args.uav_antenna * beta_uav_ue[i,j] *(kesi_qk[i,j]/(kesi_qk[i,j]+1))+args.uav_antenna*\
            (tau_p*p_k)*(beta_uav_ue[i,j]/(kesi_qk[i,j]+1))**2/\
            (p_k * tau_p * (np.sum((beta_uav_ue/(kesi_qk+1))[i, pilot_index == pilot_index[j]])) + sigma)
        channel_estimate_uav_ue[f'uav_{i+1}'] = channel_estimate_uav

    Y_aps = {}
    channel_estimate_ap_ue = {}
    for i in range(len(channel_ap_ue)): # 这么多个ap
        receive = np.zeros((len(channel_ap_ue[f'ap_{1}'][f'ue_{1}']),pilots[0].shape[-1],),dtype=np.complex128)
        for j in range(len(channel_ap_ue[f'ap_{i+1}'])): # 这么多个ue
            temp = np.sqrt(p_k * tau_p) * (channel_ap_ue[f'ap_{i+1}'][f'ue_{j+1}'][np.newaxis, :].conj().T  @  pilots [pilot_index[j]][np.newaxis,:])
            receive += temp
        noise_receive = complex_gaussian(sigma, size=receive.shape, test=test) # np.random.normal(0,10,size=receive.shape)
        Y_aps[f'ap_{i+1}'] = receive + noise_receive

    for i in range(len(channel_ap_ue)): 
        channel_estimate_ap = {}
        for j in range(len(channel_ap_ue[f'ap_{i+1}'])): # 这么多个ue
            up = np.sqrt(p_k * tau_p) * beta_ap_ue[i,j]
            down = p_k * tau_p * (np.sum(beta_ap_ue[i, pilot_index == pilot_index[j]])) + sigma
            c_mk = up/down
            channel_estimate_ap[f'ue_{j+1}'] = c_mk * (Y_aps[f'ap_{i+1}'] @ pilots[pilot_index[j]][np.newaxis,:].conj().T).conj().T
            c_ap[i,j] = c_mk
            gamma_ap[i,j] = np.sqrt(tau_p*p_k) * c_mk* beta_ap_ue[i,j] * args.ap_antenna
        channel_estimate_ap_ue[f'ap_{i+1}'] = channel_estimate_ap
        

    return channel_estimate_uav_ue, channel_estimate_ap_ue, c_uav, c_ap, gamma_uav, gamma_ap


def complex_gaussian(sigma=1.0, size = 1, test=False):
    if test:
        np.random.seed(42)
    real = np.random.normal(loc=0.0, scale=sigma / np.sqrt(2), size=size)
    imag = np.random.normal(loc=0.0, scale=sigma / np.sqrt(2), size=size)
    return real + 1j * imag