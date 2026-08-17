import numpy as np
import cvxpy as cp
from .DCPutils import get_wireless_fronthaul_noise,get_capacity
from numpy import sqrt
import math

from my_function.problem import Problem
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MPLBACKEND"] = "Agg"

best_alpha = 0.15
step = 1
def dc_power_allocation(args,
                        M, Q, K,
                        Pm_max, Pq_max,
                        g_q,
                        p_mk_prev, p_qk_prev,
                        rho_mk_prev, rho_qk_prev,
                        v_m_prev, v_q_prev,
                        sigma_k2, 
                        tau_d, tau_c,
                        pilot_index,
                        beta_mk,
                        beta_qk,
                        gamma_ap,
                        gamma_uav,
                        kesi_qk,
                        c_mk, c_qk,
                        solver='MOSEK',
                        SE_history=0,
                        alpha=0.04,
                        iteration=None
                    ):
    
    ln2 = np.log(2)
    
    cpu_uav_P_max   = np.ones((args.uav_num)) * 10**((args.power_cpu_in_dBm - 30)/10)  # W
    cpu_ap_capacity = np.ones((args.ap_num )) * args.wired_capacity  # bps/Hz
    C_G = np.sum(cpu_ap_capacity)
    sigma_mk2_prev = 1./(2**(rho_mk_prev * (v_m_prev * C_G)[:, np.newaxis]) - 1)
    C_U = np.zeros((args.uav_num,))
    N_0 = 5e-17 # 4e-21 * args.B_F * 1e6
    for i in range(args.uav_num):
        C_U[i] = get_capacity(args, channel=g_q[i], Pmax=cpu_uav_P_max[i], N0=N_0)
    
    sigma_qk2_prev = 1./(2**(rho_qk_prev * (v_q_prev * C_U)[:, np.newaxis]) - 1)
    N_G, N_U = args.ap_antenna, args.uav_antenna
    
    def compute_IN_vals(p_mk, p_qk, rho_mk, rho_qk, v_m, v_q):
        sigma_mk2 = 1./(2**(rho_mk * (v_m * C_G)[:, np.newaxis]) - 1)
        C_U_temp = np.zeros((args.uav_num,))
        for i in range(args.uav_num):
            C_U_temp[i] = get_capacity(args, channel=g_q[i], Pmax=cpu_uav_P_max[i], N0=N_0)
        sigma_qk2 = 1./(2**(rho_qk * (v_q * C_U_temp)[:, np.newaxis]) - 1)
        IN_vals = np.zeros(K)
        for k in range(K):
            BU = 0
            BU += np.sum(p_mk[:, k] * (beta_mk[:, k] + (N_G - 1 - Gamma(N_G)**2) * gamma_ap[:, k]/N_G))+\
                  np.sum(p_qk[:, k] * (beta_qk[:, k] + (N_U - 1 - Gamma(N_U)**2) * gamma_uav[:, k]/N_U))
            UI = 0
            QN = 0
            for k_ in range(K):
                for m in range(M):
                    UI += p_mk[m, k_] * (
                        beta_mk[m, k] + (N_G - 1) * gamma_ap[m, k]/N_G * (pilot_index[k] == pilot_index[k_])
                    ) * (k_ != k)
                    QN += p_mk[m, k_] * sigma_mk2[m, k_] * (
                        beta_mk[m, k] + (N_G - 1) * gamma_ap[m, k]/N_G * (pilot_index[k] == pilot_index[k_])
                    )
                for q in range(Q):
                    UI += p_qk[q, k_] * (
                        beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k]/N_U * (pilot_index[k] == pilot_index[k_])
                    ) * (k_ != k)
                    QN += p_qk[q, k_] * sigma_qk2[q, k_] * (
                        beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k]/N_U * (pilot_index[k] == pilot_index[k_])
                    )
                for m in range(M):
                    for m_ in range(M):
                        if m == m_ or pilot_index[k] != pilot_index[k_]:
                            continue
                        coef = Gamma(N_G)**2 / N_G
                        sqrt_val = np.sqrt(gamma_ap[m, k] * gamma_ap[m_, k])
                        UI += coef * np.sqrt(p_mk[m, k_] * p_mk[m_, k_]) * sqrt_val * (k_ != k)
                        QN += coef * np.sqrt(p_mk[m, k_] * p_mk[m_, k_] * sigma_mk2[m, k_] * sigma_mk2[m_, k_]) * sqrt_val
                for q in range(Q):
                    for q_ in range(Q):
                        if q == q_ or pilot_index[k] != pilot_index[k_]:
                            continue
                        coef = Gamma(N_U)**2 / N_U
                        sqrt_val = np.sqrt(gamma_uav[q, k] * gamma_uav[q_, k])
                        UI += coef * np.sqrt(p_qk[q, k_] * p_qk[q_, k_]) * sqrt_val * (k_ != k)
                        QN += coef * np.sqrt(p_qk[q, k_] * p_qk[q_, k_] * sigma_qk2[q, k_] * sigma_qk2[q_, k_]) * sqrt_val
                for m in range(M):
                    for q in range(Q):
                        if pilot_index[k] != pilot_index[k_]:
                            continue
                        coef = (Gamma(N_G) * Gamma(N_U)) / np.sqrt(N_G * N_U)
                        sqrt_val = np.sqrt(gamma_ap[m, k] * gamma_uav[q, k])
                        UI += 2* coef * np.sqrt(p_mk[m, k_] * p_qk[q, k_]) * sqrt_val * (k_ != k)
                        QN += 2* coef * np.sqrt(p_mk[m, k_] * p_qk[q, k_] * sigma_mk2[m, k_] * sigma_qk2[q, k_]) * sqrt_val
            IN_vals[k] = BU + UI + QN + sigma_k2[k]
        return IN_vals
    def compute_grad_IN(k, p_mk_num, p_qk_num, sigma_mk2_num, sigma_qk2_num):
        grad = np.zeros((M + Q, K))
        
        for m in range(M):
            grad[m, k] += beta_mk[m, k] + (N_G - 1 - Gamma(N_G)**2) * gamma_ap[m, k] / N_G
        for q in range(Q):
            grad[M + q, k] += beta_qk[q, k] + (N_U - 1 - Gamma(N_U)**2) * gamma_uav[q, k] / N_U
        
        for k_ in range(K):

            for m in range(M):
                grad[m, k_] += (
                    beta_mk[m, k] + (N_G - 1) * gamma_ap[m, k]/N_G * (pilot_index[k] == pilot_index[k_])
                ) * (k_ != k)


                grad[m, k_] += sigma_mk2_num[m, k_] * (
                    beta_mk[m, k] + (N_G - 1) * gamma_ap[m, k]/N_G * (pilot_index[k] == pilot_index[k_])
                )
            
            for q in range(Q):
                grad[M + q, k_] += (
                    beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k]/N_U * (pilot_index[k] == pilot_index[k_])
                ) * (k_ != k)
                
                grad[M + q, k_] += sigma_qk2_num[q, k_] * (
                    beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k]/N_U * (pilot_index[k] == pilot_index[k_])
                )
 
            
            for m in range(M):
                for m_ in range(M):
                    if m == m_:
                        continue
                    if pilot_index[k] != pilot_index[k_]:
                        continue
                    
                    coef = Gamma(N_G)**2 / N_G
                    sqrt_val = np.sqrt(gamma_ap[m, k] * gamma_ap[m_, k])
                    
                    if p_mk_num[m, k_] > 0 and p_mk_num[m_, k_] > 0:
                        grad[m, k_]  += coef * sqrt_val * 0.5 * np.sqrt(p_mk_num[m_, k_]/p_mk_num[m , k_]) * (k_ != k)
                        grad[m_, k_] += coef * sqrt_val * 0.5 * np.sqrt(p_mk_num[m , k_]/p_mk_num[m_, k_]) * (k_ != k)
                    
                    if (p_mk_num[m, k_] * sigma_mk2_num[m, k_] > 0 and 
                        p_mk_num[m_, k_] * sigma_mk2_num[m_, k_] > 0):
                        qn_coef = coef * np.sqrt(sigma_mk2_num[m, k_] * sigma_mk2_num[m_, k_])
                        grad[m, k_] += qn_coef * sqrt_val * 0.5 * np.sqrt(
                            p_mk_num[m_, k_] * sigma_mk2_num[m_, k_] / (p_mk_num[m, k_] * sigma_mk2_num[m, k_])
                        )
                        grad[m_, k_] += qn_coef * sqrt_val * 0.5 * np.sqrt(
                            p_mk_num[m, k_] * sigma_mk2_num[m, k_] / (p_mk_num[m_, k_] * sigma_mk2_num[m_, k_])
                        )
            
            for q in range(Q):
                for q_ in range(Q):
                    if q == q_:
                        continue
                    if pilot_index[k] != pilot_index[k_]:
                        continue
                    
                    coef = Gamma(N_U)**2 / N_U
                    sqrt_val = np.sqrt(gamma_uav[q, k] * gamma_uav[q_, k])
                    
                    if p_qk_num[q, k_] > 0 and p_qk_num[q_, k_] > 0:
                        grad[M + q , k_] += coef * sqrt_val * 0.5 * np.sqrt(p_qk_num[q_, k_]/p_qk_num[q , k_]) * (k_ != k)
                        grad[M + q_, k_] += coef * sqrt_val * 0.5 * np.sqrt(p_qk_num[q, k_] /p_qk_num[q_, k_]) * (k_ != k)
                    
                    if (p_qk_num[q, k_] * sigma_qk2_num[q, k_] > 0 and 
                        p_qk_num[q_, k_] * sigma_qk2_num[q_, k_] > 0):
                        qn_coef = coef * np.sqrt(sigma_qk2_num[q, k_] * sigma_qk2_num[q_, k_])
                        grad[M + q, k_] += qn_coef * sqrt_val * 0.5 * np.sqrt(
                            p_qk_num[q_, k_] * sigma_qk2_num[q_, k_] / (p_qk_num[q, k_] * sigma_qk2_num[q, k_])
                        )
                        grad[M + q_, k_] += qn_coef * sqrt_val * 0.5 * np.sqrt(
                            p_qk_num[q, k_] * sigma_qk2_num[q, k_] / (p_qk_num[q_, k_] * sigma_qk2_num[q_, k_])
                        )
            
            for m in range(M):
                for q in range(Q):
                    if pilot_index[k] != pilot_index[k_]:
                        continue
                    
                    coef = Gamma(N_U) * Gamma(N_G) / np.sqrt(N_G * N_U)
                    sqrt_val = np.sqrt(gamma_ap[m, k] * gamma_uav[q, k])
                    
                    if p_mk_num[m, k_] > 0 and p_qk_num[q, k_] > 0:
                        grad[m, k_]     += 2* coef * sqrt_val * 0.5 * np.sqrt(p_qk_num[q, k_]/p_mk_num[m, k_]) * (k_ != k)
                        grad[M + q, k_] += 2* coef * sqrt_val * 0.5 * np.sqrt(p_mk_num[m, k_]/p_qk_num[q, k_]) * (k_ != k)
                    
                    if (p_mk_num[m, k_] * sigma_mk2_num[m, k_] > 0 and 
                        p_qk_num[q, k_] * sigma_qk2_num[q, k_] > 0):
                        qn_coef = coef * np.sqrt(sigma_mk2_num[m, k_] * sigma_qk2_num[q, k_])
                        
                        grad[m, k_]     += 2* qn_coef * sqrt_val * 0.5 * np.sqrt(
                            p_qk_num[q, k_] * sigma_qk2_num[q, k_] / (p_mk_num[m, k_] * sigma_mk2_num[m, k_])
                        )
                        grad[M + q, k_] += 2* qn_coef * sqrt_val * 0.5 * np.sqrt(
                            p_mk_num[m, k_] * sigma_mk2_num[m, k_] / (p_qk_num[q, k_] * sigma_qk2_num[q, k_])
                        )

        return grad
    def compute_IN_vars(p_mk, p_qk, rho_mk, rho_qk, v_m, v_q):
        sigma_mk2 = 1. / (2**(rho_mk * (v_m * C_G)[:, np.newaxis]) - 1)
        C_U_temp = np.zeros((args.uav_num,))
        for i in range(args.uav_num):
            C_U_temp[i] = get_capacity(args, channel=g_q[i], Pmax=cpu_uav_P_max[i], N0=N_0)
        sigma_qk2 = 1. / (2**(rho_qk * (v_q * C_U_temp)[:, np.newaxis]) - 1)

        pilot_mask = (pilot_index[:, None] == pilot_index[None, :])
        diff_mask = ~np.eye(K, dtype=bool)

        IN_expr = []

        for k in range(K):
            # =========================
            term_mk = beta_mk[:, k] + (N_G - 1 - Gamma(N_G)**2) * gamma_ap [:, k] / N_G
            term_qk = beta_qk[:, k] + (N_U - 1 - Gamma(N_U)**2) * gamma_uav[:, k] / N_U
            
            BU = cp.sum(cp.multiply(p_mk[:, k], term_mk)) + cp.sum(cp.multiply(p_qk[:, k], term_qk))

            UI_terms = []
            QN_terms = []

            for k_ in range(K):
                is_same_pilot = float(pilot_mask[k, k_])
                is_diff_k = float(diff_mask[k, k_])

                ap_lin_coef = (beta_mk[:, k] + (N_G - 1) * gamma_ap[:, k]/N_G * is_same_pilot)
                if is_diff_k:
                    UI_terms.append(cp.sum(cp.multiply(ap_lin_coef, p_mk[:, k_])))
                QN_terms.append(cp.sum(cp.multiply(ap_lin_coef * sigma_mk2[:, k_], p_mk[:, k_])))

                uav_lin_coef = (beta_qk[:, k] + (N_U - 1) * gamma_uav[:, k]/N_U * is_same_pilot)
                if is_diff_k:
                    UI_terms.append(cp.sum(cp.multiply(uav_lin_coef, p_qk[:, k_])))
                QN_terms.append(cp.sum(cp.multiply(uav_lin_coef * sigma_qk2[:, k_], p_qk[:, k_])))

                if not is_same_pilot:
                    continue

                m_idx, m_idx_ = np.triu_indices(M, k=1) 
                ap_cross_coef = 2 * (Gamma(N_G)**2 / N_G) * \
                                np.sqrt(gamma_ap[m_idx, k] * gamma_ap[m_idx_, k])
                
                ap_geo_UI = [cp.geo_mean(cp.hstack([p_mk[m_idx[i], k_], p_mk[m_idx_[i], k_]])) for i in range(len(m_idx))]
                ap_geo_QN = [cp.geo_mean(cp.hstack([p_mk[m_idx[i], k_] * sigma_mk2[m_idx[i], k_], 
                                                    p_mk[m_idx_[i], k_] * sigma_mk2[m_idx_[i], k_]])) for i in range(len(m_idx))]
                
                if is_diff_k and len(ap_geo_UI) > 0:
                    UI_terms.append(cp.sum(cp.multiply(ap_cross_coef, cp.hstack(ap_geo_UI))))
                if len(ap_geo_QN) > 0:
                    QN_terms.append(cp.sum(cp.multiply(ap_cross_coef, cp.hstack(ap_geo_QN))))

                q_idx, q_idx_ = np.triu_indices(Q, k=1)
                uav_cross_coef = 2 * (Gamma(N_U)**2 / N_U) * \
                                np.sqrt(gamma_uav[q_idx, k] * gamma_uav[q_idx_, k])
                
                uav_geo_UI = [cp.geo_mean(cp.hstack([p_qk[q_idx[i], k_], p_qk[q_idx_[i], k_]])) for i in range(len(q_idx))]
                uav_geo_QN = [cp.geo_mean(cp.hstack([p_qk[q_idx[i], k_] * sigma_qk2[q_idx[i], k_], 
                                                     p_qk[q_idx_[i], k_] * sigma_qk2[q_idx_[i], k_]])) for i in range(len(q_idx))]
                
                if is_diff_k and len(uav_geo_UI) > 0:
                    UI_terms.append(cp.sum(cp.multiply(uav_cross_coef, cp.hstack(uav_geo_UI))))
                if len(uav_geo_QN) > 0:
                    QN_terms.append(cp.sum(cp.multiply(uav_cross_coef, cp.hstack(uav_geo_QN))))

                m_grid, q_grid = np.meshgrid(np.arange(M), np.arange(Q), indexing='ij')
                m_idx_cross = m_grid.flatten()
                q_idx_cross = q_grid.flatten()

                ap_uav_cross_coef = 2 * (Gamma(N_G)*Gamma(N_U) / np.sqrt(N_G*N_U)) * \
                                    np.sqrt(gamma_ap[m_idx_cross, k] * gamma_uav[q_idx_cross, k])
                
                ap_uav_geo_UI = [cp.geo_mean(cp.hstack([p_mk[m_idx_cross[i], k_], p_qk[q_idx_cross[i], k_]])) for i in range(len(m_idx_cross))]
                ap_uav_geo_QN = [cp.geo_mean(cp.hstack([p_mk[m_idx_cross[i], k_] * sigma_mk2[m_idx_cross[i], k_], 
                                                        p_qk[q_idx_cross[i], k_] * sigma_qk2[q_idx_cross[i], k_]])) for i in range(len(m_idx_cross))]

                if is_diff_k and len(ap_uav_geo_UI) > 0:
                    UI_terms.append(cp.sum(cp.multiply(ap_uav_cross_coef, cp.hstack(ap_uav_geo_UI))))
                if len(ap_uav_geo_QN) > 0:
                    QN_terms.append(cp.sum(cp.multiply(ap_uav_cross_coef, cp.hstack(ap_uav_geo_QN))))

            IN_expr.append(BU + cp.sum(UI_terms) + cp.sum(QN_terms) + sigma_k2[k])

        return cp.hstack(IN_expr)

    def get_DS_vec(p_mk, p_qk):
        DS_ap = np.sum((np.sqrt(p_mk * gamma_ap / N_G) * Gamma(N_G)), axis=-2)
        DS_uav= np.sum((np.sqrt(p_qk * gamma_uav / N_U) * Gamma(N_U)), axis=-2)
        DS2 = (DS_ap + DS_uav)**2
        return DS2
    
    def compute_IN_vals_vec(args,
                            p_mk, p_qk,              # (M,K), (Q,K)
                            rho_mk, rho_qk,          # (M,K), (Q,K)
                            v_m, v_q,                # (M,), (Q,)
                            gamma_ap, gamma_uav,
                            beta_mk, beta_qk,
                            ):
        M, Q, K = args.ap_num, args.uav_num, args.ue_num
        N_G, N_U = args.ap_antenna, args.uav_antenna
        gammaG2, gammaU2=Gamma(N_G)**2, Gamma(N_U)**2
        sigma_mk2 = 1.0 / ((2 ** ((rho_mk * (v_m * C_G)[:,:, None]))) - 1.0)   # (M,K)
        sigma_qk2 = 1.0 / ((2 ** (rho_qk * (v_q * C_U)[:,:, None])) - 1.0)  # (Q,K)

        pilot_index1 = pilot_index[None].repeat(step,0)
        pilot_mask = (pilot_index1[:, :, None] == pilot_index1[:, None, :]).astype(float)  # (K,K)
        eyeK = np.eye(K, dtype=float)
        diff_mask = 1.0 - eyeK

        term_ap_expect = beta_mk + (N_G - 1 - gammaG2) * gamma_ap / float(N_G)   # (M,K)
        BU_ap = np.sum(p_mk * term_ap_expect, axis=-2)  # (K,)
        term_uav_expect = beta_qk + (N_U - 1 - gammaU2) * gamma_uav / float(N_U)  # (Q,K)
        BU_uav = np.sum(p_qk * term_uav_expect, axis=-2)  # (K,)

        BU = BU_ap + BU_uav  # (K,)

        term_ap = beta_mk[:, :, :, None] + ((N_G - 1.0) / float(N_G)) * gamma_ap[:, :, :, None] * pilot_mask[:, None, :, :]  # (M,K,K)
        coef_ap_base = (p_mk[:, :, None, :])    # (M,1,K)  last axis = k' (source)
        UI_ap_mat = coef_ap_base * term_ap * diff_mask[None, None, :, :]      # (M,K,K)  multiply by (k' != k)
        UI_ap = np.sum(UI_ap_mat, axis=(1, 3))                          # (K,)

        QN_ap_mat = (p_mk[:, :, None, :] * sigma_mk2[:, :, None, :]) * term_ap  # (M,K,K)
        QN_ap = np.sum(QN_ap_mat, axis=(1, 3))                        # (K,)

        term_uav = beta_qk[:, :, :, None] + ((N_U - 1.0) / float(N_U)) * gamma_uav[:, :, :, None] * pilot_mask[:, None, :, :]  # (Q,K,K)

        coef_uav_base = (p_qk[:, :, None, :])   # (Q,1,K)
        UI_uav_mat = coef_uav_base * term_uav * diff_mask[None, None, :, :]
        UI_uav = np.sum(UI_uav_mat, axis=(1, 3))

        QN_uav_mat = (p_qk[:, :, None, :] * sigma_qk2[:, :, None, :]) * term_uav
        QN_uav = np.sum(QN_uav_mat, axis=(1, 3))

        coef_mm_kp = (gammaG2 / float(N_G))         # (M,M,K_src)
        sqrt_p_mm = np.sqrt(p_mk[:, :, None, :] * p_mk[:, None, :, :])  # (M,M,K_src)

        sqrt_gamma_mm_k = np.sqrt(gamma_ap[:, :, None, :] * gamma_ap[:, None, :, :])  # (M,M,K_dest)

        # expand to (M,M,K_dest,K_src)
        coef_mm_kp_exp = coef_mm_kp#[:, :, :, None, :]    # (M,M,1,K_src)
        sqrt_p_mm_exp = sqrt_p_mm[:, :, :, None, :]      # (M,M,1,K_src)
        sqrt_gamma_mm_k_exp = sqrt_gamma_mm_k[:, :, :, :, None]  # (M,M,K_dest,1)

        APAP_base = coef_mm_kp_exp * sqrt_p_mm_exp * sqrt_gamma_mm_k_exp  # (M,M,K_dest,K_src)

        mm_mask = (1.0 - np.eye(M, dtype=float))[None, :, :, None, None]  # exclude m==m'
        # UI: need pilot_same & k' != k
        APAP_UI_mat = APAP_base * mm_mask * pilot_mask[:, None, None, :, :] * diff_mask[None, None, None, :, :]
        UI_ap_cross = np.sum(APAP_UI_mat, axis=(1, 2, 4)) # np.sum(APAP_UI_mat, axis=(1, 2, ))[:, :, 0]   # (K_dest,) TODO

        # QN: include sigma at k'
        sqrt_sigma_mm = np.sqrt(sigma_mk2[:, :, None, :] * sigma_mk2[:, None, :, :])  # (M,M,K_src)
        sqrt_sigma_mm_exp = sqrt_sigma_mm[:, :, :, None, :]  # (M,M,1,K_src)
        APAP_QN_mat = coef_mm_kp_exp * sqrt_p_mm_exp * sqrt_sigma_mm_exp * sqrt_gamma_mm_k_exp
        APAP_QN_mat = APAP_QN_mat * mm_mask * pilot_mask[:, None, None, :, :]
        QN_ap_cross =  np.sum(APAP_QN_mat, axis=(1, 2, 4))# np.sum(APAP_QN_mat, axis=(1, 2, ))[:, :, 0]  # (K_dest,) TODO

        coef_qq_kp = (gammaU2 / float(N_U))
        sqrt_p_qq = np.sqrt(p_qk[:, :, None, :] * p_qk[:, None, :, :])
        sqrt_gamma_qq_k = np.sqrt(gamma_uav[:, :, None, :] * gamma_uav[:, None, :, :])  # (Q,Q,K_dest)

        coef_qq_kp_exp = coef_qq_kp# [:, :, :, None, :]
        sqrt_p_qq_exp = sqrt_p_qq[:, :, :, None, :]
        sqrt_gamma_qq_k_exp = sqrt_gamma_qq_k[:, :, :, :, None]

        qq_mask = (1.0 - np.eye(Q, dtype=float))[None, :, :, None, None]

        UUUU_base = coef_qq_kp_exp * sqrt_p_qq_exp * sqrt_gamma_qq_k_exp
        UUUU_UI_mat = UUUU_base * qq_mask * pilot_mask[:, None, None, :, :] * diff_mask[None, None, None, :, :]
        UI_uav_cross = np.sum(UUUU_UI_mat, axis=(1, 2, 4)) # np.sum(UUUU_UI_mat, axis=(1, 2, ))[:, :, 0] TODO

        sqrt_sigma_qq = np.sqrt(sigma_qk2[:, :, None, :] * sigma_qk2[:, None, :, :])
        sqrt_sigma_qq_exp = sqrt_sigma_qq[:, :, :, None, :]
        UUUU_QN_mat = coef_qq_kp_exp * sqrt_p_qq_exp * sqrt_sigma_qq_exp * sqrt_gamma_qq_k_exp
        UUUU_QN_mat = UUUU_QN_mat * qq_mask * pilot_mask[:, None, None, :, :]
        QN_uav_cross = np.sum(UUUU_QN_mat, axis=(1, 2, 4)) # np.sum(UUUU_QN_mat, axis=(1, 2, ))[:, :, 0] TODO

        coef_cross_factor = np.sqrt(gammaG2 * gammaU2) / np.sqrt(float(N_G * N_U))  # scalar: (Gamma(N_G)*Gamma(N_U))/sqrt(N_G*N_U)

        # build (M,Q,K_src)
        coef_mq_kp = coef_cross_factor  # (M,Q,K_src)
        sqrt_p_mq = np.sqrt(p_mk[:, :, None, :] * p_qk[:, None, :, :])               # (M,Q,K_src)

        # sqrt gamma across destination k: sqrt(gamma_ap[m,k] * gamma_uav[q,k]) -> (M,Q,K_dest)
        sqrt_gamma_mq_k = np.sqrt(gamma_ap[:, :, None, :] * gamma_uav[:, None, :, :])  # (M,Q,K_dest)

        coef_mq_kp_exp = coef_mq_kp#[:, :, :, None, :]   # (M,Q,1,K_src)
        sqrt_p_mq_exp = sqrt_p_mq[:, :, :, None, :]     # (M,Q,1,K_src)
        sqrt_gamma_mq_k_exp = sqrt_gamma_mq_k[:, :, :, :, None]  # (M,Q,K_dest,1)

        cross_base = 2.0 * coef_mq_kp_exp * sqrt_p_mq_exp * sqrt_gamma_mq_k_exp  # (M,Q,K_dest,K_src)

        # UI: need pilot_same & k' != k
        cross_UI_mat = cross_base * pilot_mask[:, None, None, :, :] * diff_mask[None, None, None, :, :]
        UI_ap_uav_cross = np.sum(cross_UI_mat, axis=(1, 2, 4))  # np.sum(cross_UI_mat, axis=(1, 2, ))[:, :, 0] TODO

        # QN: include sigma factors at k'
        sqrt_sigma_mq = np.sqrt(sigma_mk2[:, :, None, :] * sigma_qk2[:, None, :, :])  # (M,Q,K_src)
        sqrt_sigma_mq_exp = sqrt_sigma_mq[:, :, :, None]  # (M,Q,1,K_src)
        cross_QN_mat = 2.0 * coef_mq_kp_exp * np.sqrt(p_mk[:, :, None, :] * p_qk[:, None, :, :])[:, :, :, None, :] * sqrt_sigma_mq_exp * sqrt_gamma_mq_k_exp
        # cross QN: apply pilot same (original循环在进入这一段时已经检查 pilot 相等)
        cross_QN_mat = cross_QN_mat * pilot_mask[:, None, None, :, :]
        QN_ap_uav_cross = np.sum(cross_QN_mat, axis=(1, 2, 4)) # np.sum(cross_QN_mat, axis=(1, 2, ))[:, :, 0] TODO

        UI_total = UI_ap + UI_uav + UI_ap_cross  + UI_uav_cross + UI_ap_uav_cross
        QN_total = QN_ap + QN_uav + QN_ap_cross  + QN_uav_cross + QN_ap_uav_cross
        IN_vals = BU + UI_total + QN_total + sigma_k2
        IN_vals = IN_vals

        return IN_vals

    p_mk_var = cp.Variable((M, K), nonneg=True)
    p_qk_var = cp.Variable((Q, K), nonneg=True)
    scaling = tau_d / tau_c
    IN_prev_all = compute_IN_vals(
        p_mk_prev, p_qk_prev,
        rho_mk_prev, rho_qk_prev,
        v_m_prev, v_q_prev
    )
    IN_var_all = compute_IN_vars(
        p_mk_var, p_qk_var,
        rho_mk_prev, rho_qk_prev,
        v_m_prev, v_q_prev
    )
    grad_all = np.zeros((K, M + Q, K))
    for k in range(K):
        grad_all[k] = compute_grad_IN(
            k,
            p_mk_prev, p_qk_prev,
            sigma_mk2_prev, sigma_qk2_prev
        )
    def build_cross_sqrt_matrix_socp(p):
        N = p.shape[0]
        S = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    S[i][j] = p[i]
                else:
                    S[i][j] = cp.geo_mean(cp.hstack([p[i], p[j]]))
        return cp.bmat(S)
    cross_sqrt_list = []
    for kk in range(K):
        p_concat_kk = cp.hstack([
            p_mk_var[:, kk],
            p_qk_var[:, kk]
        ])
        cross_sqrt_list.append(
            build_cross_sqrt_matrix_socp(p_concat_kk)
        )
    SE_expr = 0

    for k in range(K):

        a_ap = Gamma(N_G) * np.sqrt(gamma_ap[:, k] / N_G)
        a_uav = Gamma(N_U) * np.sqrt(gamma_uav[:, k] / N_U)
        a_k = np.concatenate([a_ap, a_uav])

        p_concat_k = cp.hstack([p_mk_var[:, k], p_qk_var[:, k]])

        DS2_diag = cp.sum(cp.multiply(a_k**2, p_concat_k))

        DS2_cross = 0

        N_tot = M + Q
        for i in range(N_tot):
            for j in range(i + 1, N_tot):
                DS2_cross += 2 * a_k[i] * a_k[j] * cp.geo_mean(
                    cp.hstack([p_concat_k[i], p_concat_k[j]])
                )

        DS2_k = DS2_diag + DS2_cross

        IN_prev = IN_prev_all[k]
        IN_var = IN_prev_all[k]
        IN_expr = IN_prev

        for i in range(K):
            IN_expr += (
                cp.sum(cp.multiply(grad_all[k, :M, i], (p_mk_var[:, i] - p_mk_prev[:, i]))) +
                cp.sum(cp.multiply(grad_all[k, M:, i], (p_qk_var[:, i] - p_qk_prev[:, i])))
            )

        term1 = cp.log(IN_var + DS2_k) / ln2

        logIN_prev = cp.log(IN_prev)

        linear_term = 0
        for i in range(K):
            linear_term += (
                cp.sum(cp.multiply(grad_all[k, :M, i], (p_mk_var[:, i] - p_mk_prev[:, i]))) +
                cp.sum(cp.multiply(grad_all[k, M:, i], (p_qk_var[:, i] - p_qk_prev[:, i])))
            ) / IN_prev

        logIN_approx = (logIN_prev + linear_term) / ln2

        SE_expr += scaling * (term1 - logIN_approx)
    # ==========================================================
    # 5. CONSTRAINTS
    # ==========================================================
    constraints = []
    for m in range(M):
        constraints.append(cp.sum(p_mk_var[m, :]) <= Pm_max[m])
    for q in range(Q):
        constraints.append(cp.sum(p_qk_var[q, :]) <= Pq_max[q])
    # ==========================================================
    # 6. SOLVE
    # ==========================================================
    problem = Problem(cp.Maximize(SE_expr), constraints)
    
    try:
        problem.solve(solver=cp.MOSEK, verbose=False,
            #           mosek_params={
            #    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-9,
            #    "MSK_DPAR_INTPNT_CO_TOL_PFEAS"  : 1e-9,
            #    "MSK_DPAR_INTPNT_CO_TOL_DFEAS"  : 1e-9}
           )
    except Exception as e:
        print(f"Solve Failed: {e}")
        return p_mk_prev, p_qk_prev, SE_history, 0, 0
    
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Optimization problem not converging, state: {problem.status}")
        return p_mk_prev, p_qk_prev, SE_history, 0, 0
    
    p_mk_new = p_mk_var.value
    p_qk_new = p_qk_var.value

    best_p_mk = p_mk_prev*(1-best_alpha) + p_mk_new*best_alpha
    best_p_qk = p_qk_prev*(1-best_alpha) + p_qk_new*best_alpha
    DS_vac_batch = get_DS_vec(best_p_mk[None, :], best_p_qk[None, :])
    IN_vec_batch = compute_IN_vals_vec(args, 
                                 best_p_mk[None, :], best_p_qk[None, :], 
                                 rho_mk_prev[None], rho_qk_prev[None], 
                                 v_m_prev[None], v_q_prev[None], 
                                 gamma_ap[None], gamma_uav[None], 
                                 beta_mk[None], beta_qk[None],
    )
    user_se = np.log2(1 + DS_vac_batch / IN_vec_batch) * (tau_d/tau_c)
    current_SE = user_se.sum()

    print(f"│  p  │ Current SE : {current_SE:.4f} │")

    return best_p_mk, best_p_qk, current_SE, user_se, best_alpha

def dc_v_allocation(args, 
                    M, Q, K, 
                    Pm_max, Pq_max, 

                    g_q, 

                    p_mk_prev, p_qk_prev,
                    rho_mk_prev, rho_qk_prev, 
                    v_m_prev, v_q_prev,

                    sigma_k2, 
                    tau_d, tau_c, 

                    pilot_index,

                    beta_mk, 
                    beta_qk, 
                    gamma_ap,
                    gamma_uav, 
                    kesi_qk,
                    # c_mk, c_qk, 
                    solver='MOSEK', 
                    SE_history=0, 
                    alpha=0.8,
                    iteration=None):

    ln_2 = np.log(2)
    
    cpu_uav_P_max = np.ones((Q)) * 10**((args.power_cpu_in_dBm - 30)/10)  # W
    cpu_ap_capacity = np.ones((M)) * args.wired_capacity  # bps/Hz
    C_G = sum(cpu_ap_capacity)
    
    sigma_mk2_prev = 1./(2**(rho_mk_prev * (v_m_prev * C_G)[:, np.newaxis]) - 1)
    
    N_0 = 5e-17# 4e-21 * 1e6 * args.B_F
    C_U = np.zeros((Q,))
    for i in range(Q):
        C_U[i] = get_capacity(args, channel=g_q[i], Pmax=cpu_uav_P_max[i], N0=N_0)
    
    sigma_qk2_prev = 1./(2**(rho_qk_prev * (v_q_prev * C_U)[:, np.newaxis]) - 1)

    N_G, N_U = args.ap_antenna, args.uav_antenna
    
    v_q_var = cp.Variable((Q,), nonneg=True)


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

            UI = 0
            QN = 0
            for k_ in range(K):

                for m in range(M):
                    UI += p_mk[m, k_] * (
                        beta_mk[m, k] + (N_G - 1) * gamma_ap[m, k]/N_G * (pilot_index[k] == pilot_index[k_])
                    ) * (k_ != k)
                    QN += p_mk[m, k_] * sigma_mk2[m, k_] * (
                        beta_mk[m, k] + (N_G - 1) * gamma_ap[m, k]/N_G * (pilot_index[k] == pilot_index[k_])
                    )

                for q in range(Q):
                    UI += p_qk[q, k_] * (
                        beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k]/N_U * (pilot_index[k] == pilot_index[k_])
                    ) * (k_ != k)
                    QN += p_qk[q, k_] * sigma_qk2[q, k_] * (
                        beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k]/N_U * (pilot_index[k] == pilot_index[k_])
                    )

                for m in range(M):
                    for m_ in range(M):
                        if m == m_ or pilot_index[k] != pilot_index[k_]:
                            continue
                        coef = Gamma(N_G)**2 / N_G
                        sqrt_val = np.sqrt(gamma_ap[m, k] * gamma_ap[m_, k])
                        UI += coef * np.sqrt(p_mk[m, k_] * p_mk[m_, k_]) * sqrt_val * (k_ != k)
                        QN += coef * np.sqrt(p_mk[m, k_] * p_mk[m_, k_] * sigma_mk2[m, k_] * sigma_mk2[m_, k_]) * sqrt_val

                for q in range(Q):
                    for q_ in range(Q):
                        if q == q_ or pilot_index[k] != pilot_index[k_]:
                            continue
                        coef = Gamma(N_U)**2 / N_U
                        sqrt_val = np.sqrt(gamma_uav[q, k] * gamma_uav[q_, k])
                        UI += coef * np.sqrt(p_qk[q, k_] * p_qk[q_, k_]) * sqrt_val * (k_ != k)
                        QN += coef * np.sqrt(p_qk[q, k_] * p_qk[q_, k_] * sigma_qk2[q, k_] * sigma_qk2[q_, k_]) * sqrt_val

                for m in range(M):
                    for q in range(Q):
                        if pilot_index[k] != pilot_index[k_]:
                            continue
                        coef = (Gamma(N_G) * Gamma(N_U)) / np.sqrt(N_G * N_U)
                        sqrt_val = np.sqrt(gamma_ap[m, k] * gamma_uav[q, k])
                        UI += 2* coef * np.sqrt(p_mk[m, k_] * p_qk[q, k_]) * sqrt_val * (k_ != k)
                        QN += 2* coef * np.sqrt(p_mk[m, k_] * p_qk[q, k_] * sigma_mk2[m, k_] * sigma_qk2[q, k_]) * sqrt_val

            IN_vals[k] = BU + UI + QN + sigma_k2[k]

        return IN_vals

    def get_DS_vec  (p_mk,p_qk, ):
        DS_ap = np.sum((sqrt(p_mk * gamma_ap /N_G) * Gamma(N_G)),axis=-2)
        DS_uav= np.sum((sqrt(p_qk * gamma_uav/N_U) * Gamma(N_U)),axis=-2)
        DS2 = (DS_ap + DS_uav)**2
        return DS2

    def compute_IN_vals_vec(args,
                            p_mk, p_qk,              # (M,K), (Q,K)
                            rho_mk, rho_qk,          # (M,K), (Q,K)
                            v_m, v_q,                # (M,), (Q,)
                            gamma_ap, gamma_uav,
                            beta_mk, beta_qk,
                            ):
        M, Q, K = args.ap_num, args.uav_num, args.ue_num
        N_G, N_U = args.ap_antenna, args.uav_antenna
        gammaG2, gammaU2=Gamma(N_G)**2, Gamma(N_U)**2
        sigma_mk2 = 1.0 / ((2 ** ((rho_mk * (v_m * C_G)[:,:, None]))) - 1.0)   # (M,K)
        sigma_qk2 = 1.0 / ((2 ** (rho_qk * (v_q * C_U)[:,:, None])) - 1.0)  # (Q,K)

        # --- pilot mask ---
        pilot_index1 = pilot_index[None].repeat(step,0)
        pilot_mask = (pilot_index1[:, :, None] == pilot_index1[:, None, :]).astype(float)  # (K,K)
        eyeK = np.eye(K, dtype=float)
        diff_mask = 1.0 - eyeK   # (K,K)  -> 1 when k' != k

        term_ap_expect = beta_mk + (N_G - 1 - gammaG2) * gamma_ap / float(N_G)   # (M,K)
        BU_ap = np.sum(p_mk * term_ap_expect, axis=-2)  # (K,)
        term_uav_expect = beta_qk + (N_U - 1 - gammaU2) * gamma_uav / float(N_U)  # (Q,K)
        BU_uav = np.sum(p_qk * term_uav_expect, axis=-2)  # (K,)

        BU = BU_ap + BU_uav  # (K,)

        term_ap = beta_mk[:, :, :, None] + ((N_G - 1.0) / float(N_G)) * gamma_ap[:, :, :, None] * pilot_mask[:, None, :, :]  # (M,K,K)
        coef_ap_base = (p_mk[:, :, None, :])    # (M,1,K)  last axis = k' (source)
        UI_ap_mat = coef_ap_base * term_ap * diff_mask[None, None, :, :]      # (M,K,K)  multiply by (k' != k)
        UI_ap = np.sum(UI_ap_mat, axis=(1, 3))                          # (K,)

        QN_ap_mat = (p_mk[:, :, None, :] * sigma_mk2[:, :, None, :]) * term_ap  # (M,K,K)
        QN_ap = np.sum(QN_ap_mat, axis=(1, 3))                        # (K,)

        term_uav = beta_qk[:, :, :, None] + ((N_U - 1.0) / float(N_U)) * gamma_uav[:, :, :, None] * pilot_mask[:, None, :, :]  # (Q,K,K)

        coef_uav_base = (p_qk[:, :, None, :])   # (Q,1,K)
        UI_uav_mat = coef_uav_base * term_uav * diff_mask[None, None, :, :]
        UI_uav = np.sum(UI_uav_mat, axis=(1, 3))

        QN_uav_mat = (p_qk[:, :, None, :] * sigma_qk2[:, :, None, :]) * term_uav
        QN_uav = np.sum(QN_uav_mat, axis=(1, 3))

       
        coef_mm_kp = (gammaG2 / float(N_G))         # (M,M,K_src)
        sqrt_p_mm = np.sqrt(p_mk[:, :, None, :] * p_mk[:, None, :, :])  # (M,M,K_src)

        # gamma for destination k: sqrt(gamma_ap[m,k] * gamma_ap[m',k]) -> shape (M,M,K_dest)
        sqrt_gamma_mm_k = np.sqrt(gamma_ap[:, :, None, :] * gamma_ap[:, None, :, :])  # (M,M,K_dest)

        # expand to (M,M,K_dest,K_src)
        coef_mm_kp_exp = coef_mm_kp# [:, :, :, None, :]    # (M,M,1,K_src)
        sqrt_p_mm_exp = sqrt_p_mm[:, :, :, None, :]      # (M,M,1,K_src)
        sqrt_gamma_mm_k_exp = sqrt_gamma_mm_k[:, :, :, :, None]  # (M,M,K_dest,1)

        APAP_base = coef_mm_kp_exp * sqrt_p_mm_exp * sqrt_gamma_mm_k_exp  # (M,M,K_dest,K_src)

        mm_mask = (1.0 - np.eye(M, dtype=float))[None, :, :, None, None]  # exclude m==m'
        # UI: need pilot_same & k' != k
        APAP_UI_mat = APAP_base * mm_mask * pilot_mask[:, None, None, :, :] * diff_mask[None, None, None, :, :]
        UI_ap_cross = np.sum(APAP_UI_mat, axis=(1, 2, 4))   # (K_dest,)

        # QN: include sigma at k' (note: original循环中对 cross QN 没有 k'!=k 的乘子)
        sqrt_sigma_mm = np.sqrt(sigma_mk2[:, :, None, :] * sigma_mk2[:, None, :, :])  # (M,M,K_src)
        sqrt_sigma_mm_exp = sqrt_sigma_mm[:, :, :, None, :]  # (M,M,1,K_src)
        APAP_QN_mat = coef_mm_kp_exp * sqrt_p_mm_exp * sqrt_sigma_mm_exp * sqrt_gamma_mm_k_exp
        APAP_QN_mat = APAP_QN_mat * mm_mask * pilot_mask[:, None, None, :, :]
        QN_ap_cross = np.sum(APAP_QN_mat, axis=(1, 2, 4))  # (K_dest,)

        coef_qq_kp = (gammaU2 / float(N_U))
        sqrt_p_qq = np.sqrt(p_qk[:, :, None, :] * p_qk[:, None, :, :])
        sqrt_gamma_qq_k = np.sqrt(gamma_uav[:, :, None, :] * gamma_uav[:, None, :, :])  # (Q,Q,K_dest)

        coef_qq_kp_exp = coef_qq_kp# [:, :, :, None, :]
        sqrt_p_qq_exp = sqrt_p_qq[:, :, :, None, :]
        sqrt_gamma_qq_k_exp = sqrt_gamma_qq_k[:, :, :, :, None]

        qq_mask = (1.0 - np.eye(Q, dtype=float))[None, :, :, None, None]

        UUUU_base = coef_qq_kp_exp * sqrt_p_qq_exp * sqrt_gamma_qq_k_exp
        UUUU_UI_mat = UUUU_base * qq_mask * pilot_mask[:, None, None, :, :] * diff_mask[None, None, None, :, :]
        UI_uav_cross = np.sum(UUUU_UI_mat, axis=(1, 2, 4))

        sqrt_sigma_qq = np.sqrt(sigma_qk2[:, :, None, :] * sigma_qk2[:, None, :, :])
        sqrt_sigma_qq_exp = sqrt_sigma_qq[:, :, :, None, :]
        UUUU_QN_mat = coef_qq_kp_exp * sqrt_p_qq_exp * sqrt_sigma_qq_exp * sqrt_gamma_qq_k_exp
        UUUU_QN_mat = UUUU_QN_mat * qq_mask * pilot_mask[:, None, None, :, :]
        QN_uav_cross = np.sum(UUUU_QN_mat, axis=(1, 2, 4))

        # ----------------------------
        # AP-UAV 交叉项
        coef_cross_factor = np.sqrt(gammaG2 * gammaU2) / np.sqrt(float(N_G * N_U))  # scalar: (Gamma(N_G)*Gamma(N_U))/sqrt(N_G*N_U)

        # build (M,Q,K_src)
        coef_mq_kp = coef_cross_factor  # (M,Q,K_src)
        sqrt_p_mq = np.sqrt(p_mk[:, :, None, :] * p_qk[:, None, :, :])               # (M,Q,K_src)

        # sqrt gamma across destination k: sqrt(gamma_ap[m,k] * gamma_uav[q,k]) -> (M,Q,K_dest)
        sqrt_gamma_mq_k = np.sqrt(gamma_ap[:, :, None, :] * gamma_uav[:, None, :, :])  # (M,Q,K_dest)

        coef_mq_kp_exp = coef_mq_kp# [:, :, :, None, :]   # (M,Q,1,K_src)
        sqrt_p_mq_exp = sqrt_p_mq[:, :, :, None, :]     # (M,Q,1,K_src)
        sqrt_gamma_mq_k_exp = sqrt_gamma_mq_k[:, :, :, :, None]  # (M,Q,K_dest,1)

        cross_base = 2.0 * coef_mq_kp_exp * sqrt_p_mq_exp * sqrt_gamma_mq_k_exp  # (M,Q,K_dest,K_src)

        cross_UI_mat = cross_base * pilot_mask[:, None, None, :, :] * diff_mask[None, None, None, :, :]
        UI_ap_uav_cross = np.sum(cross_UI_mat, axis=(1, 2, 4))

        sqrt_sigma_mq = np.sqrt(sigma_mk2[:, :, None, :] * sigma_qk2[:, None, :, :])  # (M,Q,K_src)
        sqrt_sigma_mq_exp = sqrt_sigma_mq[:, :, :, None]  # (M,Q,1,K_src)
        cross_QN_mat = 2.0 * coef_mq_kp_exp * np.sqrt(p_mk[:, :, None, :] * p_qk[:, None, :, :])[:, :, :, None, :] * sqrt_sigma_mq_exp * sqrt_gamma_mq_k_exp
        cross_QN_mat = cross_QN_mat * pilot_mask[:, None, None, :, :]
        QN_ap_uav_cross = np.sum(cross_QN_mat, axis=(1, 2, 4))

        # ----------------------------
        UI_total = UI_ap + UI_uav + UI_ap_cross  + UI_uav_cross + UI_ap_uav_cross
        QN_total = QN_ap + QN_uav + QN_ap_cross  + QN_uav_cross + QN_ap_uav_cross
        IN_vals = BU + UI_total + QN_total + sigma_k2
        IN_vals = IN_vals

        return IN_vals

    def compute_grad_IN_all(p_mk, p_qk, rho_mk, rho_qk, v_m, v_q):

        Grad_D = np.zeros((K, Q))

        dN_dv_all = np.zeros((Q, K))
        for q in range(Q):
            for k_user in range(K):
                N_val = sigma_qk2_prev[q, k_user]
                coef = rho_qk_prev[q, k_user] * C_U[q] 
                dN_dv_all[q, k_user] = -1.0 * N_val * (N_val + 1.0) * ln_2 * coef

        for k in range(K):

            interfering_users = [u for u in range(K) if pilot_index[u] == pilot_index[k]]

            for q in range(Q):
                grad_val = 0.0
                
                for k_prime in interfering_users:

                    term_uav = (beta_qk[q, k] + 
                                (N_U - 1) * gamma_uav[q, k] / N_U * (k == k_prime)) # 注意 k==k' 时 gamma系数不同? 

                    factor_uav = beta_qk[q, k] + (N_U - 1) * gamma_uav[q, k] / N_U
                    
                    # d(QN_direct)/dv_q = Coef * dN/dv
                    grad_val += p_qk[q, k_prime] * dN_dv_all[q, k_prime] * factor_uav

                    # --- B. UAV-UAV 交叉项 (Cross QN) ---
                    # Term: Coef * sqrt(p1*p2) * sqrt(gamma1*gamma2) * sqrt(sigma1 * sigma2)
                    # 对 v_q 求导，涉及 q 的项可能是 sigma1 或者 sigma2
                    for q_other in range(Q):
                        if q_other == q: continue # 跳过自身，上面已算
                        
                        # 系数
                        coef_cross = Gamma(N_U)**2 / N_U
                        sqrt_p = np.sqrt(p_qk[q, k_prime] * p_qk[q_other, k_prime])
                        sqrt_gamma = np.sqrt(gamma_uav[q, k] * gamma_uav[q_other, k])
                        
                        N_q = sigma_qk2_prev[q, k_prime]
                        N_other = sigma_qk2_prev[q_other, k_prime]
                        dN_dq = dN_dv_all[q, k_prime]
                        
                        d_sqrt_sigma = np.sqrt(N_other) * (0.5 / np.sqrt(N_q)) * dN_dq
                        
                        grad_val += coef_cross * sqrt_p * sqrt_gamma * d_sqrt_sigma

                    for m in range(M):
                        coef_mq = (Gamma(N_G) * Gamma(N_U)) / np.sqrt(N_G * N_U)
                        sqrt_p_mq = np.sqrt(p_mk[m, k_prime] * p_qk[q, k_prime])
                        sqrt_gamma_mq = np.sqrt(gamma_ap[m, k] * gamma_uav[q, k])
                        
                        # sigma项: sqrt(N_m * N_q)
                        N_m = sigma_mk2_prev[m, k_prime] # AP noise, const w.r.t v_q
                        N_q = sigma_qk2_prev[q, k_prime]
                        dN_dq = dN_dv_all[q, k_prime]
                        
                        # d/dv_q (sqrt(N_m * N_q)) = sqrt(N_m) * 0.5/sqrt(N_q) * dN_q/dv_q
                        d_sqrt_sigma = np.sqrt(N_m) * (0.5 / np.sqrt(N_q)) * dN_dq
                        
                        grad_val += 2 * coef_mq * sqrt_p_mq * sqrt_gamma_mq * d_sqrt_sigma

                Grad_D[k, q] = grad_val
        
        return Grad_D

    IN_vals_current = compute_IN_vals(p_mk_prev, p_qk_prev, rho_mk_prev, rho_qk_prev, v_m_prev, v_q_prev)
    
    DS_current = np.sqrt(get_DS_vec(p_mk_prev, p_qk_prev)) # 注意：get_DS_vec 返回的是 DS^2
    Sk_vals_current = DS_current**2

    Grad_D_matrix = compute_grad_IN_all(p_mk_prev, p_qk_prev, rho_mk_prev, rho_qk_prev, v_m_prev, v_q_prev)

    
    obj_coef = np.zeros(Q)
    scaling_factor = tau_d / tau_c / ln_2
    
    for k in range(K):
        S = Sk_vals_current[k]
        D = IN_vals_current[k]
        
        chain_factor = -1.0 * S / (D * (S + D))
        
        obj_coef += scaling_factor * chain_factor * Grad_D_matrix[k, :]

    lambda_reg = 100#1
    linear_gain = cp.sum(obj_coef @ (v_q_var - v_q_prev))
    penalty = -lambda_reg * cp.sum_squares(v_q_var - v_q_prev)
    SE_proxy = linear_gain  + penalty
    SE_linear_change = cp.sum(obj_coef @ (v_q_var - v_q_prev))
    
    current_SE_val = np.sum((tau_d / tau_c) * np.log2(1 + Sk_vals_current / IN_vals_current))
    SE_expr = current_SE_val + SE_proxy

    min_bw = 1e-5
    constraints = [
        cp.sum(v_q_var) <= 1,
    ]
    
    problem = Problem(cp.Maximize(SE_expr), constraints)
    try:
        if solver == 'SCS':
            problem.solve(solver=cp.SCS, verbose=False)
        elif solver == 'ECOS':
            problem.solve(solver=cp.ECOS, verbose=False)
        elif solver == 'MOSEK':
            problem.solve(solver=cp.MOSEK, verbose=False)
    except:
        print("Solver failed, use the previous bandwidth allocation")
        return v_m_prev, v_q_prev, SE_history, None, None

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Solver failed with status {problem.status}")

        return v_m_prev, v_q_prev, SE_history, None, None

    v_q_new = v_q_var.value
    v_q_old = v_q_prev.copy()
    step = 1
    v_q_best = v_q_new*best_alpha + v_q_old*(1-best_alpha)
    DS_vac = get_DS_vec(p_mk_prev[None, :].repeat(step,0), p_qk_prev[None, :].repeat(step,0))
    IN_vec = compute_IN_vals_vec(args, 
                                 p_mk_prev[None, :], p_qk_prev[None, :], 
                                 rho_mk_prev[None], rho_qk_prev[None], 
                                 v_m_prev[None], v_q_best[None], 
                                 gamma_ap[None], gamma_uav[None], 
                                 beta_mk[None], beta_qk[None],
    )
    user_se = np.log2(1+DS_vac / IN_vec) * (tau_d / tau_c)
    current_SE = user_se.sum()
    
    print(f"│  ν  │ Current SE : {current_SE:.4f} │")
    
    return v_m_prev, v_q_best, current_SE, user_se, best_alpha

def Gamma(N):
    return math.gamma(N + 0.5) / math.gamma(N)
