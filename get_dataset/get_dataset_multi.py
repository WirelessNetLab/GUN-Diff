from my_function import lg, pg, pa, ce, gpmb
from my_function.communication.channel_geneate import channel_generate as cgp
from my_function.Algorithm.DCPutils import get_capacity as gc
from my_function import get_expert_dcp # get_expert_wmmse, get_expert_sca, 
from get_args import get_args

import torch

# from my_function.communication.data_transmit import data_in_BS
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os
# from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
os.environ["MPLBACKEND"] = "Agg"


def get_dataset(args, test):
    dataset = {}
    SE = np.zeros((args.buffer_size))
    # Create Buffer
    channel_batch        = {}

    p_G_batch            = {}
    p_U_batch            = {}
    v_q_batch            = {}

    location_batch       = {}

    beta_batch           = {}

    precode_batch        = {}
    

    compress_noise_batch = {}

    sigma_k_sq_batch     = {}
    user_history_batch   = {}

    SE_history           = {}

    dataset              = {}

    pilot_index_batch    = {}

    gamma_ap_batch       = {}
    gamma_uav_batch      = {}
    kesi_qk_batch        = {}
    c_mk_batch           = {}
    c_qk_batch           = {}

    # pbar = tqdm(range(args.buffer_size))
    # for i in pbar:
    for i in range(args.buffer_size):
        start_time = datetime.now()
        location_index = lg(args, test)
        # ds(location_index, args.block_lenth)

        channel, beta, _, kesi_qk = cgp(args, location_index, test)
        pilots = pg(args)
        pilot_index, pilot_allocation = pa(pilots, args.ue_num, test)
        channel_estimate_uav_ue, channel_estimate_ap_ue, c_uav, c_ap, gamma_uav, gamma_ap = \
            ce(args, channel=channel, pilots=pilots, pilot_index=pilot_index, beta=beta, p_k=0.2, tau_p = args.tau_p, kesi_qk=kesi_qk, test=test)

        precode_matrix_uav, precode_matrix_ap = gpmb(channel_estimate_uav_ue, channel_estimate_ap_ue, method = args.method)

        # precode_matrix_uav, precode_matrix_ap = gpmb(channel['uav_ue'], channel['ap_ue'])

        precode = {
                    'uav' : precode_matrix_uav,
                    'ap'  : precode_matrix_ap
                    }
    
        N_0 = 2e-18
        Pm = np.ones((args.ap_num)) * 10 ** (args.Pm_ap -30)
        Pq = np.ones((args.uav_num)) * 10 ** (args.Pm_uav -30)
        sigma_k_sq = np.ones((args.ue_num))  * N_0  

        if args.solver == 'DCP':
            result = get_expert_dcp(args = args,
                                    M=args.ap_num, Q=args.uav_num, K=args.ue_num, 
                                    Pm_max=Pm, Pq_max=Pq,
                                    precode=precode,       # shape: (M, K, N_m), (Q, K, N_q)
                                    channel=channel,       # shape: (M, K, N_m), (Q, K, N_q)
                                    sigma_k2=sigma_k_sq, 

                                    pilot_index=pilot_index,
                                    beta=beta,
                                    gamma_ap=gamma_ap,gamma_uav=gamma_uav,
                                    kesi_qk=kesi_qk,
                                    c_ap=c_ap,c_uav=c_uav,

                                    tau_d=args.tau_d, tau_c=args.tau_c,
                                    max_iter=args.max_iter,
                                    correct = True if args.f > 2.0 else False
                                    )#, tol=1e-3)
        
        p_mk, p_qk  = result['p_mk'], result['p_qk']
        v_q         = result['v_q']
        SE[i]       = result['SE']
        SE_history1 = result['SE_history']
        user_history= result['user_se']

        channel_batch           [f'data_{i+1}'] = channel
        precode_batch           [f'data_{i+1}'] = precode
        # compress_noise_batch    [f'data_{i+1}'] = compress_noise

        beta_batch              [f'data_{i+1}'] = beta
        p_G_batch               [f'data_{i+1}'] = p_mk
        p_U_batch               [f'data_{i+1}'] = p_qk
        v_q_batch               [f'data_{i+1}'] = v_q

        location_batch          [f'data_{i+1}'] = location_index 
        sigma_k_sq_batch        [f'data_{i+1}'] = sigma_k_sq
        SE_history              [f'data_{i+1}'] = SE_history1
        user_history_batch      [f'data_{i+1}'] = user_history

        pilot_index_batch       [f'data_{i+1}'] = pilot_index
        gamma_ap_batch          [f'data_{i+1}'] = gamma_ap
        gamma_uav_batch         [f'data_{i+1}'] = gamma_uav
        kesi_qk_batch           [f'data_{i+1}'] = kesi_qk
        c_mk_batch              [f'data_{i+1}'] = c_ap
        c_qk_batch              [f'data_{i+1}'] = c_uav
        end_time = datetime.now()
        

        print(f'Epoch {i+1}/{args.buffer_size} : Dateset Got Finished, Use Time:{end_time - start_time}\n')

    #   ###################################################################################################
    dataset['channel'],dataset['precode'],dataset['compress_noise'] = channel_batch, precode_batch, compress_noise_batch
    dataset['beta'],dataset['p_G'],dataset['p_U'], dataset['v_q'] = beta_batch, p_G_batch, p_U_batch, v_q_batch
    dataset['location'] = location_batch
    dataset['receive_noise'] = sigma_k_sq_batch
    dataset['SE'] = SE          # SE
    dataset['SE_history'] = SE_history
    dataset['user_se'   ] = user_history_batch

    dataset['pilot_index'] = pilot_index_batch 
    dataset['gamma_ap'  ]  = gamma_ap_batch    
    dataset['gamma_uav' ]  = gamma_uav_batch   
    dataset['kesi_qk'   ]  = kesi_qk_batch     
    dataset['c_ap'      ]  = c_mk_batch        
    dataset['c_uav'     ]  = c_qk_batch        



    channel_get, precode_get, compress_noise_get = dataset['channel'],dataset['precode'],dataset['compress_noise']
    beta_get, p_G_get, p_U_get, v_U_get = dataset['beta'],dataset['p_G'],dataset['p_U'], dataset['v_q']
    location__get = dataset['location'      ]
    rec_noise_get = dataset['receive_noise' ]
    SE_expert_get = torch.tensor(dataset['SE'],dtype=torch.float32,device=args.device)
    history       = dataset['SE_history'    ]
    ue_se_history = dataset['user_se']

    pilot_index__get = dataset['pilot_index'   ]
    gamma_ap__get    = dataset['gamma_ap'      ]
    gamma_uav__get   = dataset['gamma_uav'     ]
    kesi_qk__get     = dataset['kesi_qk'       ]
    c_mk__get        = dataset['c_ap'          ]
    c_qk__get        = dataset['c_uav'         ]
    
    pilot_index_get = torch.zeros((args.buffer_size, args.ue_num), dtype = torch.float32, device=args.device)
    ue_se_get       = torch.zeros((args.buffer_size, args.ue_num), dtype = torch.float32, device=args.device)
    gamma_ap_get    = torch.zeros((args.buffer_size, args.ap_num , args.ue_num), dtype = torch.float32, device=args.device)
    gamma_uav_get   = torch.zeros((args.buffer_size, args.uav_num, args.ue_num), dtype = torch.float32, device=args.device)
    kesi_qk_get     = torch.zeros((args.buffer_size, args.uav_num, args.ue_num), dtype = torch.float32, device=args.device)
    c_mk_get        = torch.zeros((args.buffer_size, args.ap_num , args.ue_num), dtype = torch.float32, device=args.device)
    c_qk_get        = torch.zeros((args.buffer_size, args.uav_num, args.ue_num), dtype = torch.float32, device=args.device)

    channel_uav_ue_get  = torch.zeros((args.buffer_size, args.uav_num, args.ue_num , args.ue_antenna, args.uav_antenna ), dtype = torch.float32, device=args.device)
    channel_ap_ue_get   = torch.zeros((args.buffer_size, args.ap_num , args.ue_num , args.ue_antenna, args.ap_antenna  ), dtype = torch.float32, device=args.device)
    channel_cpu_uav_get = torch.zeros((args.buffer_size, 1           , args.ap_num , args.ap_antenna, args.cpu_antenna ), dtype = torch.float32, device=args.device)

    precode_uav_ue_get  = torch.zeros((args.buffer_size, args.uav_num, args.ue_num, args.uav_antenna, args.ue_antenna  ), dtype = torch.float32, device=args.device)
    precode_ap_ue_get   = torch.zeros((args.buffer_size, args.ap_num , args.ue_num, args.ap_antenna , args.ue_antenna  ), dtype = torch.float32, device=args.device)


    compress_noise_uav_get = torch.zeros((args.buffer_size, args.uav_num,), dtype = torch.float32, device=args.device)
    compress_noise_ap_get  = torch.zeros((args.buffer_size, args.ap_num ,), dtype = torch.float32, device=args.device)

    beta_uav_ue_get = torch.zeros((args.buffer_size, args.uav_num, args.ue_num), dtype = torch.float32, device=args.device)
    beta_ap_ue_get  = torch.zeros((args.buffer_size, args.ap_num , args.ue_num), dtype = torch.float32, device=args.device)

    p_UAV_get = torch.zeros((args.buffer_size, args.uav_num , args.ue_num ), dtype = torch.float32, device=args.device)
    p_AP_get  = torch.zeros((args.buffer_size, args.ap_num  , args.ue_num ), dtype = torch.float32, device=args.device)
    v_UAV_get = torch.zeros((args.buffer_size, args.cpu_num , args.uav_num), dtype = torch.float32, device=args.device)
    

    receive_noise_get = torch.zeros((args.buffer_size, args.ue_num))

    location_uav_get  = torch.zeros((args.buffer_size, args.uav_num,3,), dtype = torch.float32, device=args.device )
    location_ap_get   = torch.zeros((args.buffer_size, args.ap_num ,3,), dtype = torch.float32, device=args.device )
    location_ue_get   = torch.zeros((args.buffer_size, args.ue_num ,3,), dtype = torch.float32, device=args.device )
    location_cpu_get  = torch.zeros((args.buffer_size, 1           ,3,), dtype = torch.float32, device=args.device )


    for i in range(args.buffer_size):
        # for c in range(args.cpu_num):
        pilot_index_get[i] = torch.tensor(pilot_index__get  [f'data_{i+1}'],dtype=torch.float32,device=args.device)
        ue_se_get      [i] = torch.tensor(ue_se_history     [f'data_{i+1}'],dtype=torch.float32,device=args.device)
        gamma_ap_get   [i] = torch.tensor(gamma_ap__get     [f'data_{i+1}'],dtype=torch.float32,device=args.device)
        gamma_uav_get  [i] = torch.tensor(gamma_uav__get    [f'data_{i+1}'],dtype=torch.float32,device=args.device)
        kesi_qk_get    [i] = torch.tensor(kesi_qk__get      [f'data_{i+1}'],dtype=torch.float32,device=args.device)
        c_mk_get       [i] = torch.tensor(c_mk__get         [f'data_{i+1}'],dtype=torch.float32,device=args.device)
        c_qk_get       [i] = torch.tensor(c_qk__get         [f'data_{i+1}'],dtype=torch.float32,device=args.device)
        for q in range(args.uav_num):
            channel_cpu_uav_get[i,0,q,] = torch.tensor(channel_get[f'data_{i+1}']['cpu_uav'][f'uav_{q+1}'],dtype=torch.float32,device=args.device)
        v_UAV_get          [i,] = torch.tensor(v_U_get[f'data_{i+1}'],dtype=torch.float32,device=args.device)
        for q in range(args.uav_num):
            for k in range(args.ue_num):
                channel_uav_ue_get[i,q,k] = torch.tensor(channel_get[f'data_{i+1}']['uav_ue'][f'uav_{q+1}'][f'ue_{k+1}'],dtype=torch.float32,device=args.device)
                precode_uav_ue_get[i,q,k] = torch.tensor(precode_get[f'data_{i+1}']['uav'][f'uav_{q+1}'][f'ue_{k+1}'],dtype=torch.float32,device=args.device)
            # compress_noise_uav_get[i,q] = torch.tensor(compress_noise_get[f'data_{i+1}']['uav'][f'uav_{q+1}'],dtype=torch.float32,device=args.device)
        beta_uav_ue_get[i,] = torch.tensor(beta_get[f'data_{i+1}']['uav_ue'],dtype=torch.float32,device=args.device)
        p_UAV_get[i,] = torch.tensor(p_U_get[f'data_{i+1}'],dtype=torch.float32,device=args.device)
        for m in range(args.ap_num):
            for k in range(args.ue_num):
                channel_ap_ue_get[i,m,k] = torch.tensor(channel_get[f'data_{i+1}']['ap_ue'][f'ap_{m+1}'][f'ue_{k+1}'],dtype=torch.float32,device=args.device)
                precode_ap_ue_get[i,m,k] = torch.tensor(precode_get[f'data_{i+1}']['ap'][f'ap_{m+1}'][f'ue_{k+1}'],dtype=torch.float32,device=args.device)
            # compress_noise_ap_get[i,m] = torch.tensor(compress_noise_get[f'data_{i+1}']['ap'][f'ap_{m+1}'],dtype=torch.float32,device=args.device)
        beta_ap_ue_get[i,] = torch.tensor(beta_get[f'data_{i+1}']['ap_ue'],dtype=torch.float32,device=args.device)
        p_AP_get[i,] = torch.tensor(p_G_get[f'data_{i+1}'],dtype=torch.float32,device=args.device)
        receive_noise_get[i,] = torch.tensor(rec_noise_get[f'data_{i+1}'],dtype=torch.float32,device=args.device)
        
        location_uav_get [i,] = torch.tensor(location__get [f'data_{i+1}']  [ 'uav' ],dtype=torch.float32,device=args.device)
        location_ap_get  [i,] = torch.tensor(location__get [f'data_{i+1}']  [ 'ap'  ],dtype=torch.float32,device=args.device)
        location_ue_get  [i,] = torch.tensor(location__get [f'data_{i+1}']  [ 'ue'  ],dtype=torch.float32,device=args.device)
        location_cpu_get [i,] = torch.tensor(location__get [f'data_{i+1}']  [ 'cpu' ],dtype=torch.float32,device=args.device)
        location_get = {
                        'uav' : location_uav_get    ,
                        'ap'  : location_ap_get     ,
                        'ue'  : location_ue_get     ,
                        'cpu' : location_cpu_get    ,
                        }

    C_U = torch.zeros((args.buffer_size, args.uav_num,),device=args.device)
    N_0 = 2e-18
    cpu_uav_P_max = torch.ones((args.buffer_size, 1, args.uav_num)) * 10**((args.power_cpu_in_dBm - 30)/10)
    for i in range(args.buffer_size):
        for j in range(args.uav_num):
            C_U[i,j] = torch.tensor(gc(args, channel=channel_cpu_uav_get[i,0,j,], Pmax=cpu_uav_P_max[i,0,j].numpy(), N0=N_0))
    C_G = torch.ones((args.buffer_size, args.ap_num),device=args.device) * args.wired_capacity # sum(cpu_ap_capacity)

    data = {
                'channel_uav_ue'    : channel_uav_ue_get     ,
                'channel_ap_ue'     : channel_ap_ue_get      ,
                'channel_cpu_uav'   : channel_cpu_uav_get    ,

                'precode_uav_ue'    : precode_uav_ue_get     ,
                'precode_ap_ue'     : precode_ap_ue_get      ,
                'compress_noise_uav': compress_noise_uav_get ,
                'compress_noise_ap' : compress_noise_ap_get  ,
                'p_UAV'             : p_UAV_get              ,
                'p_AP'              : p_AP_get               ,
                'v_UAV'             : v_UAV_get              ,
                'receive_noise'     : receive_noise_get      ,
                'SE_expert'         : SE_expert_get          ,
                'location'          : location_get           ,
                'agrs'              : args                   ,
                'beta_uav'          : beta_uav_ue_get        ,
                'beta_ap'           : beta_ap_ue_get         ,
                'SE_history'        : history                ,
                'ue_SE_history'     : ue_se_get              ,

                'pilot_index'       : pilot_index_get        ,
                'gamma_ap'          : gamma_ap_get           ,
                'gamma_uav'         : gamma_uav_get          ,
                'kesi_qk'           : kesi_qk_get            ,
                'c_mk'              : c_mk_get               ,
                'c_qk'              : c_qk_get               ,
                'C_G'               : C_G                    ,
                'C_U'               : C_U                    ,
            }
    now = datetime.now()
    data_path = os.path.join('dataset', f'{now.strftime("%Y_%m_%d_%H_%M_%S")}_{args.buffer_size}_dataset.pt')

    torch.save(data, data_path)
    print('DataSet Saved Finished')







if __name__ == '__main__':
    args = get_args()
    get_dataset(args, test=False)