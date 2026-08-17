import numpy as np
import cvxpy as cp
from datetime import datetime
from .DCPutils import get_capacity

from .DCP_single import Gamma,dc_power_allocation, dc_v_allocation
from get_args import get_args

from numpy import log2, sqrt

import os
os.environ["MPLBACKEND"] = "Agg"

def Multiple_optm(args,
                        M, Q, K,
                        Pm_max, Pq_max,
                        precode,channel,
                        sigma_k2, 
                        pilot_index,
                        beta,
                        gamma_ap, gamma_uav,
                        kesi_qk,
                        c_ap, c_uav,

                        tau_d, tau_c,
                                     #  p       v     
                        solver      =['MOSEK','MOSEK' ],#['SCS', 'SCS'],
                        optm_choice =[True   , True   ],
                        alpha       =[0.05, 0.0005],
                        dec_cof     =[1.0    , 1.0    ],
                        max_iter=30, tol=1e-4,
                        
                        correct=True):


    SE_history        = []
    user_se_history   = []

    p_mk_history      = []
    p_qk_history      = []

    v_q_history       = []
    v_m_history       = []

    rho_qk_history    = []
    rho_mk_history    = []

    seta_q_history    = []
    alpha_list        = []
    alpha_v_list      = []

    beta_mk = beta['ap_ue']
    beta_qk = beta['uav_ue']
    c_mk, c_qk = c_ap, c_uav,



    # 数据预处理
    w_qk, w_mk, g_qk, g_mk, g_q  = data_pretreat(precode, channel=channel) #


    # Initial Power Allocation ##############################################
    p_mk = np.ones((M, K)) * max(Pm_max.mean() / K, 1e-3)
    p_qk = np.ones((Q, K)) * max(Pq_max.mean() / K, 1e-3)

    # Initial Bandwidth Allocation ##############################################
    v_q = np.ones((Q,)) / Q

    for m in range(M):
        channel_gains = np.sum(np.abs(g_mk[m])**2, axis=1)  # (K,)
        p_mk[m] = Pm_max[m] * channel_gains / np.sum(channel_gains)

    for q in range(Q):
        channel_gains = np.sum(np.abs(g_qk[q])**2, axis=1)  # (K,)
        p_qk[q] = Pq_max[q] * channel_gains / np.sum(channel_gains)
    
    # ###############################
    
    v_m = np.ones((args.ap_num,))/args.ap_num
    # ###############################
    rho_qk   = np.ones((args.uav_num, args.ue_num)) /args.ue_num
    rho_mk   = np.ones((args.ap_num , args.ue_num)) /args.ue_num
    # ###############################
    seta_q = np.ones((args.uav_num,)) * 90 # °
    # end #########################################################
    # p_mk = np.zeros((M, K))
    # p_qk = np.zeros((Q, K))
    # v_q  = np.ones((Q,)) / Q
    # end end #########################################################

    p_mk_prev   = p_mk.copy()
    p_qk_prev   = p_qk.copy()

    rho_mk_prev = rho_mk.copy()
    rho_qk_prev = rho_qk.copy()

    v_q_prev    = v_q.copy()
    v_m_prev    = v_m.copy()

    seta_q_prev = seta_q.copy()

    prev_obj = -np.inf

    cpu_uav_P_max = np.ones((args.uav_num)) * 10**((args.power_cpu_in_dBm - 30)/10)  # W
    cpu_ap_capacity = np.ones((args.ap_num)) * args.wired_capacity  # bps/Hz

    C_G = sum(cpu_ap_capacity)
    sigma_mk2_prev = 1./(2**(rho_mk_prev * (v_m_prev * C_G)[:, np.newaxis].repeat(repeats=K, axis=-1)) - 1)
    N_G, N_U = args.ap_antenna, args.uav_antenna
    N_0 = 2e-16


    def compute_IN_vals(p_mk, p_qk, rho_mk, rho_qk, v_m, v_q):

        sigma_mk2 = 1./(2**(rho_mk * (v_m * C_G)[:, np.newaxis]) - 1)

        C_U_temp = np.zeros((args.uav_num,))
        for i in range(args.uav_num):
            C_U_temp[i] = get_capacity(args, channel=g_q[i], Pmax=cpu_uav_P_max[i], N0=N_0)
        sigma_qk2 = 1./(2**(rho_qk * (v_q * C_U_temp)[:, np.newaxis]) - 1)

        IN_vals = np.zeros(K)
        for k in range(K):
            # ========== 波束项 BU ==========
            BU = 0
            BU += np.sum(p_mk[:, k] * (beta_mk[:, k] + (N_G - 1 - Gamma(N_G)**2) * gamma_ap[:, k]/N_G))+\
                  np.sum(p_qk[:, k] * (beta_qk[:, k] + (N_U - 1 - Gamma(N_U)**2) * gamma_uav[:, k]/N_U))

            # for m in range(M):
            #     BU += p_mk[m, k] * (beta_mk[m, k] + (N_G - 1 - Gamma(N_G)**2) * gamma_ap[m, k]/N_G)
            # for q in range(Q):
            #     BU += p_qk[q, k] * (beta_qk[q, k]/(kesi_qk[q, k] + 1) + (N_U - 1 - Gamma(N_U)**2) * gamma_uav[q, k]/N_U)

            # ========== 干扰与压缩噪声项 UI + QN ==========
            UI = 0
            QN = 0
            for k_ in range(K):
                # if k_ == k:
                #     continue  # 跳过自身用户

                # ---------- 1. AP 内部 ----------
                for m in range(M):
                    UI += p_mk[m, k_] * (
                        beta_mk[m, k] + (N_G - 1) * gamma_ap[m, k]/N_G * (pilot_index[k] == pilot_index[k_])
                    ) * (k_ != k)
                    QN += p_mk[m, k_] * sigma_mk2[m, k_] * (
                        beta_mk[m, k] + (N_G - 1) * gamma_ap[m, k]/N_G * (pilot_index[k] == pilot_index[k_])
                    )

                # ---------- 2. UAV 内部 ----------
                for q in range(Q):
                    UI += p_qk[q, k_] * (
                        beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k]/N_U * (pilot_index[k] == pilot_index[k_])
                    ) * (k_ != k)
                    QN += p_qk[q, k_] * sigma_qk2[q, k_] * (
                        beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k]/N_U * (pilot_index[k] == pilot_index[k_])
                    )

                # ---------- 3. AP-AP 交叉 ----------
                for m in range(M):
                    for m_ in range(M):
                        if m == m_ or pilot_index[k] != pilot_index[k_]:
                            continue
                        coef = Gamma(N_G)**2 / N_G
                        sqrt_val = np.sqrt(gamma_ap[m, k] * gamma_ap[m_, k])
                        UI += coef * np.sqrt(p_mk[m, k_] * p_mk[m_, k_]) * sqrt_val * (k_ != k)
                        QN += coef * np.sqrt(p_mk[m, k_] * p_mk[m_, k_] * sigma_mk2[m, k_] * sigma_mk2[m_, k_]) * sqrt_val

                # ---------- 4. UAV-UAV 交叉 ----------
                for q in range(Q):
                    for q_ in range(Q):
                        if q == q_ or pilot_index[k] != pilot_index[k_]:
                            continue
                        coef = Gamma(N_U)**2 / N_U
                        sqrt_val = np.sqrt(gamma_uav[q, k] * gamma_uav[q_, k])
                        UI += coef * np.sqrt(p_qk[q, k_] * p_qk[q_, k_]) * sqrt_val * (k_ != k)
                        QN += coef * np.sqrt(p_qk[q, k_] * p_qk[q_, k_] * sigma_qk2[q, k_] * sigma_qk2[q_, k_]) * sqrt_val

                # ---------- 5. AP-UAV 交叉 ----------
                for m in range(M):
                    for q in range(Q):
                        if pilot_index[k] != pilot_index[k_]:
                            continue
                        # 交叉干扰
                        coef = (Gamma(N_G) * Gamma(N_U)) / np.sqrt(N_G * N_U)
                        sqrt_val = np.sqrt(gamma_ap[m, k] * gamma_uav[q, k])
                        UI += 2* coef * np.sqrt(p_mk[m, k_] * p_qk[q, k_]) * sqrt_val * (k_ != k)
                        # 交叉压缩噪声
                        QN += 2* coef * np.sqrt(p_mk[m, k_] * p_qk[q, k_] * sigma_mk2[m, k_] * sigma_qk2[q, k_]) * sqrt_val

            IN_vals[k] = BU + UI + QN + sigma_k2[k]

        return IN_vals
 
    # 向量化加速代码
    def compute_IN_vals_vec(
                            p_mk, p_qk,              # (M,K), (Q,K)
                            rho_mk, rho_qk,          # (M,K), (Q,K)
                            v_m, v_q,                # (M,), (Q,)
                            ):
        gammaG2, gammaU2=Gamma(N_G)**2, Gamma(N_U)**2
        # --- sigma 计算 ---
        sigma_mk2 = 1.0 / (2 ** (rho_mk * (v_m * C_G)[:, None]) - 1.0)   # (M,K)

        # C_U_temp: 短 loop（Q 通常较小）
        uav_num = args.uav_num
        C_U_temp = np.zeros((uav_num,))
        for i in range(uav_num):
            C_U_temp[i] = get_capacity(channel=g_q[i], Pmax=cpu_uav_P_max[i], N0=N_0)
        sigma_qk2 = 1.0 / (2 ** (rho_qk * (v_q * C_U_temp)[:, None]) - 1.0)  # (Q,K)

        # --- pilot mask ---
        pilot_index1 = np.asarray(pilot_index)   # (K,)
        pilot_mask = (pilot_index1[:, None] == pilot_index1[None, :]).astype(float)  # (K,K)
        eyeK = np.eye(K, dtype=float)
        diff_mask = 1.0 - eyeK   # (K,K)  -> 1 when k' != k

        # --- BU (期望信号) ---
        term_ap_expect = beta_mk + (N_G - 1 - gammaG2) * gamma_ap / float(N_G)   # (M,K)
        BU_ap = np.sum(p_mk * term_ap_expect, axis=0)  # (K,)

        term_uav_expect = beta_qk / (kesi_qk + 1.0) + (N_U - 1 - gammaU2) * gamma_uav / float(N_U)  # (Q,K)
        BU_uav = np.sum(p_qk * term_uav_expect, axis=0)  # (K,)

        BU = BU_ap + BU_uav  # (K,)

        # ----------------------------
        # AP 内部 UI, QN
        # term_ap[m,k,k'] = beta_mk[m,k] + (N_G-1)/N_G * gamma_ap[m,k] * pilot_mask[k,k']
        term_ap = beta_mk[:, :, None] + ((N_G - 1.0) / float(N_G)) * gamma_ap[:, :, None] * pilot_mask[None, :, :]  # (M,K,K)

        coef_ap_base = (p_mk[:, None, :])    # (M,1,K)  last axis = k' (source)
        UI_ap_mat = coef_ap_base * term_ap * diff_mask[None, :, :]      # (M,K,K)  multiply by (k' != k)
        UI_ap = np.sum(UI_ap_mat, axis=(0, 2))                          # (K,)

        QN_ap_mat = (p_mk[:, None, :] * sigma_mk2[:, None, :]) * term_ap  # (M,K,K)
        QN_ap = np.sum(QN_ap_mat, axis=(0, 2))                        # (K,)

        # ----------------------------
        # UAV 内部 UI, QN (同理)
        term_uav = beta_qk[:, :, None] / (kesi_qk + 1.0) + ((N_U - 1.0) / float(N_U)) * gamma_uav[:, :, None] * pilot_mask[None, :, :]  # (Q,K,K)

        coef_uav_base = (p_qk[:, None, :])   # (Q,1,K)
        UI_uav_mat = coef_uav_base * term_uav * diff_mask[None, :, :]
        UI_uav = np.sum(UI_uav_mat, axis=(0, 2))

        QN_uav_mat = (p_qk[:, None, :] * sigma_qk2[:, None, :]) * term_uav
        QN_uav = np.sum(QN_uav_mat, axis=(0, 2))

        # ----------------------------
        # AP-AP 交叉 (m != m', 需同 pilot)
        # build (M,M,K_src) arrays for c and p over k' (source index)
        coef_mm_kp = (gammaG2 / float(N_G))         # (M,M,K_src)
        sqrt_p_mm = np.sqrt(p_mk[:, None, :] * p_mk[None, :, :])  # (M,M,K_src)

        # gamma for destination k: sqrt(gamma_ap[m,k] * gamma_ap[m',k]) -> shape (M,M,K_dest)
        sqrt_gamma_mm_k = np.sqrt(gamma_ap[:, None, :] * gamma_ap[None, :, :])  # (M,M,K_dest)

        # expand to (M,M,K_dest,K_src)
        coef_mm_kp_exp = coef_mm_kp# [:, :, None, :]    # (M,M,1,K_src)
        sqrt_p_mm_exp = sqrt_p_mm[:, :, None, :]      # (M,M,1,K_src)
        sqrt_gamma_mm_k_exp = sqrt_gamma_mm_k[:, :, :, None]  # (M,M,K_dest,1)

        APAP_base = coef_mm_kp_exp * sqrt_p_mm_exp * sqrt_gamma_mm_k_exp  # (M,M,K_dest,K_src)

        mm_mask = (1.0 - np.eye(M, dtype=float))[:, :, None, None]  # exclude m==m'
        # UI: need pilot_same & k' != k
        APAP_UI_mat = APAP_base * mm_mask * pilot_mask[None, None, :, :] * diff_mask[None, None, :, :]
        UI_ap_cross = np.sum(APAP_UI_mat, axis=(0, 1, 3))   # (K_dest,)

        # QN: include sigma at k' (note: original循环中对 cross QN 没有 k'!=k 的乘子)
        sqrt_sigma_mm = np.sqrt(sigma_mk2[:, None, :] * sigma_mk2[None, :, :])  # (M,M,K_src)
        sqrt_sigma_mm_exp = sqrt_sigma_mm[:, :, None, :]  # (M,M,1,K_src)
        APAP_QN_mat = coef_mm_kp_exp * sqrt_p_mm_exp * sqrt_sigma_mm_exp * sqrt_gamma_mm_k_exp
        APAP_QN_mat = APAP_QN_mat * mm_mask * pilot_mask[None, None, :, :]
        QN_ap_cross = np.sum(APAP_QN_mat, axis=(0, 1, 3))  # (K_dest,)

        # ----------------------------
        # UAV-UAV 交叉 (同理)

        coef_qq_kp = (gammaU2 / float(N_U))
        sqrt_p_qq = np.sqrt(p_qk[:, None, :] * p_qk[None, :, :])
        sqrt_gamma_qq_k = np.sqrt(gamma_uav[:, None, :] * gamma_uav[None, :, :])  # (Q,Q,K_dest)

        coef_qq_kp_exp = coef_qq_kp# [:, :, None, :]
        sqrt_p_qq_exp = sqrt_p_qq[:, :, None, :]
        sqrt_gamma_qq_k_exp = sqrt_gamma_qq_k[:, :, :, None]

        qq_mask = (1.0 - np.eye(Q, dtype=float))[:, :, None, None]

        UUUU_base = coef_qq_kp_exp * sqrt_p_qq_exp * sqrt_gamma_qq_k_exp
        UUUU_UI_mat = UUUU_base * qq_mask * pilot_mask[None, None, :, :] * diff_mask[None, None, :, :]
        UI_uav_cross = np.sum(UUUU_UI_mat, axis=(0, 1, 3))

        sqrt_sigma_qq = np.sqrt(sigma_qk2[:, None, :] * sigma_qk2[None, :, :])
        sqrt_sigma_qq_exp = sqrt_sigma_qq[:, :, None, :]
        UUUU_QN_mat = coef_qq_kp_exp * sqrt_p_qq_exp * sqrt_sigma_qq_exp * sqrt_gamma_qq_k_exp
        UUUU_QN_mat = UUUU_QN_mat * qq_mask * pilot_mask[None, None, :, :]
        QN_uav_cross = np.sum(UUUU_QN_mat, axis=(0, 1, 3))

        # ----------------------------
        # AP-UAV 交叉项
        coef_cross_factor = np.sqrt(gammaG2 * gammaU2) / np.sqrt(float(N_G * N_U))  # scalar: (Gamma(N_G)*Gamma(N_U))/sqrt(N_G*N_U)

        # build (M,Q,K_src)
        coef_mq_kp = coef_cross_factor  # (M,Q,K_src)
        sqrt_p_mq = np.sqrt(p_mk[:, None, :] * p_qk[None, :, :])               # (M,Q,K_src)

        # sqrt gamma across destination k: sqrt(gamma_ap[m,k] * gamma_uav[q,k]) -> (M,Q,K_dest)
        sqrt_gamma_mq_k = np.sqrt(gamma_ap[:, None, :] * gamma_uav[None, :, :])  # (M,Q,K_dest)

        coef_mq_kp_exp = coef_mq_kp# [:, :, None, :]   # (M,Q,1,K_src)
        sqrt_p_mq_exp = sqrt_p_mq[:, :, None, :]     # (M,Q,1,K_src)
        sqrt_gamma_mq_k_exp = sqrt_gamma_mq_k[:, :, :, None]  # (M,Q,K_dest,1)

        cross_base = 2.0 * coef_mq_kp_exp * sqrt_p_mq_exp * sqrt_gamma_mq_k_exp  # (M,Q,K_dest,K_src)

        # UI: need pilot_same & k' != k
        cross_UI_mat = cross_base * pilot_mask[None, None, :, :] * diff_mask[None, None, :, :]
        UI_ap_uav_cross = np.sum(cross_UI_mat, axis=(0, 1, 3))

        # QN: include sigma factors at k'
        sqrt_sigma_mq = np.sqrt(sigma_mk2[:, None, :] * sigma_qk2[None, :, :])  # (M,Q,K_src)
        sqrt_sigma_mq_exp = sqrt_sigma_mq[:, :, None, :]  # (M,Q,1,K_src)
        cross_QN_mat = 2.0 * coef_mq_kp_exp * np.sqrt(p_mk[:, None, :] * p_qk[None, :, :])[:, :, None, :] * sqrt_sigma_mq_exp * sqrt_gamma_mq_k_exp
        cross_QN_mat = cross_QN_mat * pilot_mask[None, None, :, :]
        QN_ap_uav_cross = np.sum(cross_QN_mat, axis=(0, 1, 3))

        # ----------------------------
        UI_total = UI_ap + UI_uav + UI_ap_cross + UI_uav_cross + UI_ap_uav_cross
        QN_total = QN_ap + QN_uav + QN_ap_cross + QN_uav_cross + QN_ap_uav_cross

        IN_vals = BU + UI_total + QN_total + sigma_k2
        IN_vals = IN_vals

        return IN_vals


    IN_vals_real = compute_IN_vals(p_mk_prev, p_qk_prev, rho_mk_prev, rho_qk_prev, v_m_prev, v_q_prev)

    DS_ap = np.sum(sqrt(p_mk_prev * gamma_ap /N_G) * Gamma(N_G),axis=0)
    DS_uav= np.sum(sqrt(p_qk_prev * gamma_uav/N_U) * Gamma(N_U),axis=0)
    DS = (DS_ap + DS_uav)**2
    current_SE = sum(log2(1+DS/IN_vals_real))
    user_se = []
    for i in range(args.ue_num):
        user_se.append(log2(1+DS[i]/IN_vals_real[i]))
    ## #####################################################################################################

    prev_obj = (tau_d / tau_c) * current_SE
    print(f'{prev_obj}\n')

    SE_history    .append(prev_obj   )
    user_se_history.append(user_se   )
    p_mk_history  .append(p_mk_prev  )
    p_qk_history  .append(p_qk_prev  )
    v_q_history   .append(v_q_prev   )
    v_m_history   .append(v_m_prev   )
    rho_qk_history.append(rho_qk_prev)
    rho_mk_history.append(rho_mk_prev)
    seta_q_history.append(seta_q_prev)

    start_time = datetime.now()
    for iteration in range(max_iter): # 交替迭代
        prev_obj = current_SE

        print('┌─────┬──────────────────────┐')
        if optm_choice[0]:
            
            p_mk_prev, p_qk_prev, current_SE, user_se, alpha_p = dc_power_allocation(args,
                                                                M, Q, K,
                                                                Pm_max, Pq_max,

                                                                g_q,

                                                                p_mk_prev,p_qk_prev,
                                                                rho_mk_prev,rho_qk_prev,
                                                                v_m_prev, v_q_prev,

                                                                sigma_k2,
                                                                tau_d,tau_c,

                                                                pilot_index,
                                                                beta_mk,
                                                                beta_qk,
                                                                gamma_ap,
                                                                gamma_uav,
                                                                kesi_qk,
                                                                c_mk,c_qk,

                                                                solver[0],
                                                                SE_history[-1],
                                                                alpha=alpha[0] * dec_cof[0],
                                                                # iteration=iteration,
                                                                )
            alpha_list     .append(alpha_p)
            SE_history     .append(  current_SE   )
            user_se_history.append(  user_se      )
            p_mk_history   .append(  p_mk_prev    )
            p_qk_history   .append(  p_qk_prev    )
            
            rho_qk_history .append(  rho_qk_prev  )
            rho_mk_history .append(  rho_mk_prev  )

            v_m_history    .append(  v_m_prev     )
            v_q_history    .append(  v_q_prev     )

            seta_q_history.append(   seta_q_prev  )

        if optm_choice[1]:
            print('├─────┼──────────────────────┤')
            v_m_prev, v_q_prev, current_SE, user_se, alpha_v = dc_v_allocation   (args, 
                                                                    M, Q, K, 
                                                                    Pm_max, Pq_max, 
                                                                    g_q, 
                                                                    p_mk_prev, p_qk_prev,
                                                                    rho_mk_prev, rho_qk_prev, 
                                                                    v_m_prev, v_q_prev,

                                                                    sigma_k2, 
                                                                    tau_d, tau_c, 
                                                                    pilot_index,
                                                                    beta_mk, beta_qk, 
                                                                    gamma_ap, gamma_uav, 
                                                                    kesi_qk,
                                                                    c_mk, c_qk, 
                                                                    solver[1],
                                                                    SE_history[-1],
                                                                    alpha=alpha[1] * dec_cof[1],
                                                                    iteration=iteration,
                                                                )

            alpha_v_list   .append(alpha_v      )
            SE_history     .append(current_SE)
            user_se_history.append(user_se)

            p_mk_history   .append(  p_mk_prev    )
            p_qk_history   .append(  p_qk_prev    )
            
            rho_qk_history .append(  rho_qk_prev  )
            rho_mk_history .append(  rho_mk_prev  )

            v_m_history    .append(  v_m_prev     )
            v_q_history    .append(  v_q_prev     )

            seta_q_history.append(   seta_q_prev  )
        print(f'└─────┴──────────────────────┘\n')
        print(f'Epoch {iteration} / {int(max_iter)} Finished\n')
        if abs((current_SE - prev_obj).item()) < tol:# or current_SE < prev_obj:
            prev_obj = current_SE
            break

    max_SE          = max(SE_history)
    index           = SE_history.index(max_SE)

    p_mk_prev       = p_mk_history[index]
    p_qk_prev       = p_qk_history[index]

    rho_mk_prev     = rho_mk_history[index]
    rho_qk_prev     = rho_qk_history[index]

    v_m_prev        = v_m_history[index]
    v_q_prev        = v_q_history[index]


    seta_q_prev     = seta_q_history[index]

    user_se_prev    = user_se_history[index]

    current_SE = max_SE

    end_time = datetime.now() 

    print(f'Problem Solve Finished, Use time: {end_time-start_time}')

    return {'p_mk'      :p_mk_prev, 

            'p_qk'      :p_qk_prev, 

            'rho_mk'    :rho_mk_prev, 

            'rho_qk'    :rho_qk_prev, 

            'v_m'       :v_m_prev, 
            'v_q'       :v_q_prev, 

            'seta_q'    :seta_q_prev,
            
            'SE'        :current_SE,
            'SE_history':SE_history,
            'user_se'   :user_se_prev,
            }


def data_pretreat(precode, channel):
    """
    """

    precode_uav, precode_ap = precode['uav'], precode['ap']
    precode_uav_matrix  = np.zeros(  (len(precode_uav), len(precode_uav['uav_1']), precode_uav['uav_1']['ue_1'].shape[0]), 
                                   dtype=np.complex128)
    precode_ap_matrix   = np.zeros(  (len(precode_ap), len(precode_ap['ap_1']), precode_ap['ap_1']['ue_1'].shape[0]), 
                                   dtype=np.complex128)
    for i in range(len(precode_uav)):
        for j in range(len(precode_uav['uav_1'])):
            precode_uav_matrix[i,j,:] = precode_uav[f'uav_{i+1}'][f'ue_{j+1}'].squeeze(-1)

    for i in range(len(precode_ap)):
        for j in range(len(precode_ap['ap_1'])):
            precode_ap_matrix[i,j,:] = precode_ap[f'ap_{i+1}'][f'ue_{j+1}'].squeeze(-1)

    channel_uav_ue, channel_ap_ue, channel_cpu_uav = channel['uav_ue'], channel['ap_ue'], channel['cpu_uav']
    channel_uav_matrix = np.zeros(  (len(channel_uav_ue), len(channel_uav_ue['uav_1']), channel_uav_ue['uav_1']['ue_1'].shape[-1]), 
                                   dtype=np.complex128)
    channel_ap_matrix  = np.zeros(  (len(channel_ap_ue), len(channel_ap_ue['ap_1']), channel_ap_ue['ap_1']['ue_1'].shape[-1]), 
                                   dtype=np.complex128)
    channel_cpu_matrix = np.zeros(  (len(channel_cpu_uav), channel_cpu_uav['uav_1'].shape[-2], channel_cpu_uav['uav_1'].shape[-1]), 
                                   dtype=np.complex128) 
    
    for i in range(len(channel_uav_ue)):
        for j in range(len(channel_uav_ue['uav_1'])):
            channel_uav_matrix[i,j,:] = channel_uav_ue[f'uav_{i+1}'][f'ue_{j+1}'].squeeze(0)

    for i in range(len(channel_ap_ue)):
        for j in range(len(channel_ap_ue['ap_1'])):
            channel_ap_matrix[i,j,:] = channel_ap_ue[f'ap_{i+1}'][f'ue_{j+1}']

    for i in range(len(channel_cpu_uav)):
            channel_cpu_matrix[i,:,:] = channel_cpu_uav[f'uav_{i+1}']
    
    
    return  precode_uav_matrix, precode_ap_matrix,\
            channel_uav_matrix, channel_ap_matrix, channel_cpu_matrix\



