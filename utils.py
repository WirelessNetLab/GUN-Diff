import numpy as np
import torch
from numpy import sqrt
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.functional import softplus


def correct_layer(args, power, v):
    if len(power.shape) !=3:
        power = power.reshape(args.batch_size, args.ap_num+args.uav_num, args.ue_num)
    if len(v.shape) !=2:
        v = v.reshape(args.batch_size, args.uav_num)
    eps = 1e-12

    power = softplus(power)
    v     = softplus(v)

    power_sum_ap = power.sum(dim=-1)

    mask_ap = power_sum_ap > 1.0  # [B, M]

    if mask_ap.any():
        norm_factor = power_sum_ap.unsqueeze(-1) + eps
        power = torch.where(
            mask_ap.unsqueeze(-1),
            power / norm_factor,
            power
        )

    v_sum = v.sum(dim=-1)

    mask_v = v_sum > 1.0

    if mask_v.any():
        v = torch.where(
            mask_v.unsqueeze(-1),
            v / (v_sum.unsqueeze(-1) + eps),
            v
        )
    return power, v



def get_DS(args,
           power,
           dataset,
           i, ):
    """
    Compute the Desired Strength
    """
    N_G, N_U = args.ap_antenna, args.uav_antenna
    power[power<0]=0
    DS_ap = np.sum((sqrt(power[:,:args.ap_num] * dataset['gamma_ap' ][i] /N_G) * Gamma(N_G)).numpy(),axis=-2)
    DS_uav= np.sum((sqrt(power[:,args.ap_num:] * dataset['gamma_uav'][i] /N_U) * Gamma(N_U)).numpy(),axis=-2)
    DS2 = (DS_ap + DS_uav)**2
    return DS2


def compute_IN_vals_vec(args,
                        p_mk, p_qk,
                        rho_mk, rho_qk,
                        v_m, v_q,
                        dataset,
                        disturb_idx,
                        ):
    """
    Compute the strength of Interference and Noise
    """
    p_mk[p_mk<0]=0
    p_qk[p_qk<0]=0
    beta_mk   = dataset['beta_ap']      .numpy()[disturb_idx]
    beta_qk   = dataset['beta_uav']     .numpy()[disturb_idx]
    gamma_ap  = dataset['gamma_ap']     .numpy()[disturb_idx]
    gamma_uav = dataset['gamma_uav']    .numpy()[disturb_idx]
    C_G       = dataset['C_G']          .numpy()[disturb_idx]
    C_U       = dataset['C_U']          .numpy()[disturb_idx]
    kesi_qk   = dataset['kesi_qk']      .numpy()[disturb_idx]
    c_ap      = dataset['c_mk']         .numpy()[disturb_idx]
    c_uav     = dataset['c_qk']        .numpy()[disturb_idx]
    sigma_k2  = dataset['receive_noise'].numpy()[disturb_idx]

    M, Q, K = args.ap_num, args.uav_num, args.ue_num
    N_G, N_U = args.ap_antenna, args.uav_antenna
    gammaG2, gammaU2=Gamma(N_G)**2, Gamma(N_U)**2

    sigma_mk2 = 1.0 / (2 ** (rho_mk * (C_G)      [:,:, None]) - 1.0)
    sigma_qk2 = 1.0 / (2 ** (rho_qk * (v_q * C_U)[:,:, None]) - 1.0)

    pilot_index1 = np.asarray(dataset['pilot_index'].numpy()[disturb_idx])
    pilot_mask = (pilot_index1[:, :, None] == pilot_index1[:, None, :]).astype(float)
    eyeK = np.eye(K, dtype=float)
    diff_mask = 1.0 - eyeK

    term_ap_expect = beta_mk + (N_G - 1 - gammaG2) * gamma_ap / float(N_G)   # (M,K)
    BU_ap = np.sum(p_mk * term_ap_expect, axis=-2)  # (K,)
    term_uav_expect = beta_qk / (kesi_qk + 1.0) + (N_U - 1 - gammaU2) * gamma_uav / float(N_U)  # (Q,K)
    BU_uav = np.sum(p_qk * term_uav_expect, axis=-2)  # (K,)

    BU = BU_ap + BU_uav  # (K,)

    # ----------------------------
    term_ap = beta_mk[:, :, :, None] + ((N_G - 1.0) / float(N_G)) * gamma_ap[:, :, :, None] * pilot_mask[:, None, :, :]  # (M,K,K)
    coef_ap_base = (p_mk[:, :, None, :])    # (M,1,K)  last axis = k' (source)
    UI_ap_mat = coef_ap_base * term_ap * diff_mask[None, None, :, :]      # (M,K,K)  multiply by (k' != k)
    UI_ap = np.sum(UI_ap_mat, axis=(1, 3))                          # (K,)

    QN_ap_mat = (p_mk[:, :, None, :] * sigma_mk2[:, :, None, :]) * term_ap  # (M,K,K)
    QN_ap = np.sum(QN_ap_mat, axis=(1, 3))                        # (K,)

    # ----------------------------
    term_uav = beta_qk[:, :, :, None] + ((N_U - 1.0) / float(N_U)) * gamma_uav[:, :, :, None] * pilot_mask[:, None, :, :]  # (Q,K,K)
    coef_uav_base = (p_qk[:, :, None, :])   # (Q,1,K)
    UI_uav_mat = coef_uav_base * term_uav * diff_mask[None, None, :, :]
    UI_uav = np.sum(UI_uav_mat, axis=(1, 3))
    QN_uav_mat = (p_qk[:, :, None, :] * sigma_qk2[:, :, None, :]) * term_uav
    QN_uav = np.sum(QN_uav_mat, axis=(1, 3))

    # ----------------------------
    coef_mm_kp = (gammaG2 / float(N_G))         # (M,M,K_src)
    sqrt_p_mm = np.sqrt(p_mk[:, :, None, :] * p_mk[:, None, :, :])  # (M,M,K_src)

    sqrt_gamma_mm_k = np.sqrt(gamma_ap[:, :, None, :] * gamma_ap[:, None, :, :])  # (M,M,K_dest)

    coef_mm_kp_exp = coef_mm_kp# [:, :, :, None, :]    # (M,M,1,K_src)
    sqrt_p_mm_exp = sqrt_p_mm[:, :, :, None, :]      # (M,M,1,K_src)
    sqrt_gamma_mm_k_exp = sqrt_gamma_mm_k[:, :, :, :, None]  # (M,M,K_dest,1)

    APAP_base = coef_mm_kp_exp * sqrt_p_mm_exp * sqrt_gamma_mm_k_exp  # (M,M,K_dest,K_src)

    mm_mask = (1.0 - np.eye(M, dtype=float))[None, :, :, None, None]  # exclude m==m'
    # UI: need pilot_same & k' != k
    APAP_UI_mat = APAP_base * mm_mask * pilot_mask[:, None, None, :, :] * diff_mask[None, None, None, :, :]
    UI_ap_cross = np.sum(APAP_UI_mat, axis=(1, 2, 4))   # (K_dest,)

    sqrt_sigma_mm = np.sqrt(sigma_mk2[:, :, None, :] * sigma_mk2[:, None, :, :])  # (M,M,K_src)
    sqrt_sigma_mm_exp = sqrt_sigma_mm[:, :, :, None, :]  # (M,M,1,K_src)
    APAP_QN_mat = coef_mm_kp_exp * sqrt_p_mm_exp * sqrt_sigma_mm_exp * sqrt_gamma_mm_k_exp
    APAP_QN_mat = APAP_QN_mat * mm_mask * pilot_mask[:, None, None, :, :]
    QN_ap_cross = np.sum(APAP_QN_mat, axis=(1, 2, 4))  # (K_dest,)

    # ----------------------------
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
    coef_cross_factor = np.sqrt(gammaG2 * gammaU2) / np.sqrt(float(N_G * N_U))

    # build (M,Q,K_src)
    coef_mq_kp = coef_cross_factor  # (M,Q,K_src)
    sqrt_p_mq = np.sqrt(p_mk[:, :, None, :] * p_qk[:, None, :, :])               # (M,Q,K_src)

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
    UI_total = UI_ap + UI_uav + UI_ap_cross + UI_uav_cross + UI_ap_uav_cross
    QN_total = QN_ap + QN_uav + QN_ap_cross + QN_uav_cross + QN_ap_uav_cross

    IN_vals = BU + UI_total + QN_total + sigma_k2
    eps = 1e-19
    IN_vals = np.maximum(IN_vals, eps)

    return IN_vals


def Gamma(N):
    return math.gamma(N + 0.5) / math.gamma(N)

class MyLrScheduler(_LRScheduler):
    def __init__(self, optimizer, schedule, base_lr=0.0, last_epoch=-1):
        self.schedule = schedule
        self.base_lr = base_lr
        self.segments = []
        self._build_segments()
        super().__init__(optimizer, last_epoch)

    def _build_segments(self):
        cur_lr = self.base_lr
        cur_step = 0
        i = 0

        while i < len(self.schedule):
            steps = self.schedule[i]
            cfg = self.schedule[i + 1]
            i += 2

            # ───── Linear → Hold ─────
            if len(cfg) == 2:
                target_lr, hold_steps = cfg

                start = cur_step
                end = cur_step + steps
                self.segments.append({
                    "start": start,
                    "end": end,
                    "fn": self._linear_fn(cur_lr, target_lr, start, steps)
                })
                cur_step = end
                cur_lr = target_lr

                if hold_steps > 0:
                    start = cur_step
                    end = cur_step + hold_steps
                    self.segments.append({
                        "start": start,
                        "end": end,
                        "fn": lambda step, lr=cur_lr: lr
                    })
                    cur_step = end

            # ───── Linear → Cosine Cycle ─────
            elif len(cfg) == 3:
                max_lr, min_lr, cycle_steps = cfg

                start = cur_step
                end = cur_step + steps
                self.segments.append({
                    "start": start,
                    "end": end,
                    "fn": self._linear_fn(cur_lr, max_lr, start, steps)
                })
                cur_step = end
                cur_lr = max_lr

                self.segments.append({
                    "start": cur_step,
                    "end": float("inf"),
                    "fn": self._cosine_cycle_fn(cur_lr, min_lr, cycle_steps, cur_step)
                })
                break

            else:
                raise ValueError("Invalid schedule config")

    def _linear_fn(self, start_lr, end_lr, start_step, steps):
        def fn(step):
            t = step - start_step
            return start_lr + (end_lr - start_lr) * t / steps
        return fn

    def _cosine_cycle_fn(self, lr_max, lr_min, cycle_steps, start_step, gamma=3.0):
        def fn(step):
            t = (step - start_step) % cycle_steps
            u = t / cycle_steps
            u = u ** gamma
            return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(2 * math.pi * u))
        return fn


    def get_lr(self):
        step = self.last_epoch + 1

        for seg in self.segments:
            if seg["start"] <= step < seg["end"]:
                lr = seg["fn"](step)
                return [lr for _ in self.optimizer.param_groups]

        return [self.base_lr for _ in self.optimizer.param_groups]
