import numpy as np
from numpy import pi, sin, cos, abs,arctan,sqrt

"""
较上一版，UAV变为多天线
"""
def channel_generate(args, locationa_index: dict, test, **kwargs):
    """
    信道生成函数

    Args:
        locationa_index : location index
        antenna_setting : antenna

    Returns:
        tuple:
            - channel : channel
            - beta : large scale fading
    """
    if test:
        np.random.seed(42)

    ap_index  = locationa_index ['ap' ]
    uav_index = locationa_index ['uav']
    ue_index  = locationa_index ['ue' ]
    cpu_index = locationa_index ['cpu']

    

    beat0_in_dB = -50 #dB
    beta0 = 10**(beat0_in_dB/10)
    a = 2.2
    kesi_1, kesi_2 = args.xi_1, args.xi_2 # 11.95, 0.136

    delta_d_C = 3e+8 /(2* args.f_F * 1e+6)  # CPU antenna spacing
    delta_d_U = 3e+8 /(2* args.f   * 1e+6)  # UAV antenna spacing

    lambda_A = 3e+8 / args.f     # Access link carrier wavelength
    lambda_F = 3e+8 / args.f_F   # Fronthaul link carrier wavelength
    
    # 1 Fronthaul CPU --> AAP
    beta_cpu_uav = np.zeros((1,uav_index.shape[0],))
    channel_cpu_uav = {}
    for i in range(uav_index.shape[0]):
        D_cpu_uav = sqrt(np.sum((uav_index[i,:] - cpu_index) ** 2,axis=1))
        d_x_y_z = abs(uav_index[i,:] - cpu_index)

        AoA  = arctan(sqrt(d_x_y_z[:,0]**2 + d_x_y_z[:,1]**2)/d_x_y_z[:,2]) *180/np.pi
        seta = arctan(d_x_y_z[:,2]/sqrt(d_x_y_z[:,0]**2 + d_x_y_z[:,1]**2)) *180/np.pi
        AoD = seta
        
        P_LOS  = 1./(1. + kesi_1 * np.exp(-kesi_2 * (seta-kesi_1)))
        P_NLOS = 1. - P_LOS
        kesi = P_LOS / P_NLOS
        beta_cpu_uav[:,i] = (beta0 * (D_cpu_uav)**(-a))

        large_shadow = sqrt(beta_cpu_uav[:,i])

        h_LOS_cpu = []
        for j in range(args.cpu_antenna):
            h_LOS_cpu.append((np.exp((-1j)*(2*np.pi * j * delta_d_C * np.sin(AoD))/lambda_F)).item())
        h_LOS_uav = []
        for j in range(args.uav_antenna):
            h_LOS_uav.append(np.exp((-1j)*(2*np.pi * j * delta_d_U * np.sin(AoA))/lambda_F).item())
        
        h_LOS_cpu, h_LOS_uav = np.array(h_LOS_cpu)[np.newaxis,:], np.array(h_LOS_uav)[np.newaxis,:]
        h_LOS = h_LOS_uav.conj().T @ h_LOS_cpu
        h_NLOS = np.random.normal(loc=0.0, scale=1.0 / sqrt(2), size=h_LOS.shape) \
               + np.random.normal(loc=0.0, scale=1.0 / sqrt(2), size=h_LOS.shape) * 1j 
        small_shadow = sqrt(kesi/(kesi+1)) * h_LOS + sqrt(1/(kesi+1)) * h_NLOS
        channel_cpu_uav[f'uav_{i+1}'] = large_shadow * small_shadow


    channel_uav_ue = {}
    channel_uav    = {}

    seta_qk        = np.zeros((uav_index.shape[0],ue_index.shape[0],))
    beta_uav_ue    = np.zeros((uav_index.shape[0],ue_index.shape[0],))
    kesi_qk        = np.zeros((uav_index.shape[0],ue_index.shape[0],))

    # 2.1 Access Link AAP --> UE
    for i in range(locationa_index ['uav'].shape[0]):
        channel_uav = {}
        for j in range(locationa_index ['ue' ].shape[0]):
            
            uav_location = uav_index[i]
            ue_location = ue_index[j]

            # TODO : 大尺度衰落建模
            d_uav_ue = abs(uav_location - ue_location)
            D_uav_ue = sqrt(np.sum((d_uav_ue) ** 2))
            seta = arctan(d_uav_ue[2]/sqrt(d_uav_ue[0]**2 + d_uav_ue[1]**2)) *180/np.pi # ° # UE抬头离地角
            seta_qk1 = arctan(sqrt(d_uav_ue[0]**2 + d_uav_ue[1]**2)/d_uav_ue[2]) *180/np.pi # °
            AoD = 90-seta # 仰角
            P_LOS = 1./(1. + kesi_1 * np.exp(-kesi_2 * (seta-kesi_1)))
            P_NLOS = 1. - P_LOS
            kesi = P_LOS/P_NLOS
            beta_uav_ue_temp =  beta0 * ((D_uav_ue)**(-a))
            
            large_shadow = sqrt(beta_uav_ue_temp)

            h_LOS_uav = []
            for k in range(args.uav_antenna):
                h_LOS_uav.append(np.exp((-1j)*(2*np.pi * k * delta_d_C * np.sin(AoD))/lambda_A).item())
            h_LOS_uav = np.array(h_LOS_uav)[np.newaxis,:]
            h_NLOS = complex_gaussian_array(h_LOS_uav.shape, sigma=1.0, test=test) # antenna_setting['antenna']['uav']
            small_shadow = sqrt(kesi/(kesi+1)) * h_LOS_uav + sqrt(1/(kesi+1)) * h_NLOS

            channel_uav [f'ue_{j+1}'] = large_shadow * small_shadow
            beta_uav_ue [i,j] = beta_uav_ue_temp

            seta_qk     [i,j] = seta_qk1
            kesi_qk     [i,j] = kesi
        channel_uav_ue  [f'uav_{i+1}'] = channel_uav

    
    # Access Link GAP --> UE
    channel_ap_ue = {}
    channel_ap = {}
    beta_ap_ue = np.zeros((ap_index.shape[0],ue_index.shape[0],))
    for i in range(locationa_index ['ap'].shape[0]):
        channel_ap = {}
        for j in range(locationa_index ['ue' ].shape[0]):
            ap = ap_index[i]
            ue = ue_index[j]
            d_slot = abs(ap - ue)
            d = sqrt(d_slot[0]**2 + d_slot[1]**2)
            f = args.f
            PL_in_dB = triple_slope_model(d,f) # dB
            PL = 10 ** (PL_in_dB/10)
            sigma_sh = 8 #dB
            z_mk = double_components_model(test=test)
            beta_ap_ue_temp = PL * (10 ** (sigma_sh * z_mk / 10))
            large_shadow = sqrt(beta_ap_ue_temp)

            small_shadow = complex_gaussian_array(args.ap_antenna, sigma=1.0, test=test)

            channel = large_shadow * small_shadow
            channel_ap[f'ue_{j+1}'] = channel
            beta_ap_ue[i,j] = beta_ap_ue_temp
        channel_ap_ue[f'ap_{i+1}'] = channel_ap
    channel = {}
    channel['cpu_uav'], channel['uav_ue'], channel['ap_ue'] = channel_cpu_uav, channel_uav_ue, channel_ap_ue
    beta = {}
    beta['cpu_uav'], beta['uav_ue'], beta['ap_ue'] = beta_cpu_uav, beta_uav_ue, beta_ap_ue
    return channel, beta, seta_qk, kesi_qk
























def complex_gaussian_array(size, sigma=1.0, test=False):
    if test:
        np.random.seed(42)
    real = np.random.normal(loc=0.0, scale=sigma / sqrt(2), size=size)
    imag = np.random.normal(loc=0.0, scale=sigma / sqrt(2), size=size)
    return real + 1j * imag

def triple_slope_model(d, f, d_1=50., d_0=10., h_ap=15., h_u=1.65):
    """
    Args:
        d    : 2D location diff
        d_1  : gate min
        d_0  : gate max
        f    : carrier in MHz
        h_ap : AP antenna height
        h_u  : UE antenna height

    Returns:
        PL : Path Loss
    """

    L = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_ap)\
    -(1.1 * np.log10(f)-0.7) * h_u + (1.56 * np.log10(f)-0.8)
    if d > d_1:
        PL_in_dB = -L - 35 * np.log10(d *1e-3)
    elif d_0 < d <= d_1:
        PL_in_dB = -L - 15 * np.log10(d_1 *1e-3) - 20 * np.log10(d *1e-3)
    elif d <=d_0:
        PL_in_dB = -L - 15 * np.log10(d_1 *1e-3) - 20 * np.log10(d_0 *1e-3)
    return PL_in_dB

def double_components_model(delta=0.5, test=False):
    if test:
        np.random.seed(42)
    a_m = np.random.randn()
    b_k = np.random.randn()
    return sqrt(delta) * a_m + sqrt(1 - delta) * b_k
