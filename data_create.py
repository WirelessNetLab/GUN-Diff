import torch
from torch_geometric.data import HeteroData
from torch.nn import functional as F
from torch import long, float64
from get_args import get_args


def data_create(args, dataset)->HeteroData:
    p_UAV               = dataset [ 'p_UAV'              ]       [:args.buffer_size].to(args.device)
    p_AP                = dataset [ 'p_AP'               ]       [:args.buffer_size].to(args.device)
    receive_noise       = dataset [ 'receive_noise'      ]       [:args.buffer_size].to(args.device)

    beta_ap             = dataset [ 'beta_ap'           ]       [:args.buffer_size].to(args.device)
    beta_uav            = dataset [ 'beta_uav'          ]       [:args.buffer_size].to(args.device)

    v_q                 = dataset [ 'v_UAV'             ]       [:args.buffer_size].to(args.device)
    C_U                 = dataset [ 'C_U'               ]       [:args.buffer_size].to(args.device)
    C_G                 = dataset [ 'C_G'               ]       [:args.buffer_size].to(args.device)
    pilot_index         = dataset ['pilot_index'        ]       [:args.buffer_size].to(args.device)
    if len(v_q.shape) !=3:
        v_q = v_q.unsqueeze(-2)
    v_m = torch.ones((args.buffer_size, 1, args.ap_num))
    pilot_num           = args.tau_p
    pilot_onehot = F.one_hot(pilot_index.to(long), num_classes=pilot_num)

    clean_p_AP  = p_AP 
    clean_p_UAV = p_UAV
    clean_v     = v_q  


    beta = torch.cat((beta_ap, beta_uav),dim=-2)
    data = HeteroData()
    Pm = torch.ones((args.buffer_size, args.ap_num , 1),device=args.device) * 10 ** (args.Pm_ap  - 30) 
    Pq = torch.ones((args.buffer_size, args.uav_num, 1),device=args.device) * 10 ** (args.Pm_uav - 30) 
    cpu_uav_P_max = torch.ones((args.buffer_size, 1, args.uav_num)) * 10**((args.power_cpu_in_dBm - 30)/10)

    C   = torch.cat((C_G, C_U),dim=-1)

    choose = False 
    # Feature of APs
    data['AP'].x        = torch.cat((v_m, v_q),dim=-1).transpose(-1, -2)   
    data['AP'].P_max    = (torch.cat((Pm  , Pq   ), dim=-2)/args.eps).log1p() if choose else torch.cat((Pm  , Pq   ), dim=-2)*args.cof
    data['AP'].x, data['AP'].P_max = data['AP'].x.to(float64), data['AP'].P_max.to(float64)

    # Feature of UEs
    data['CPU'].x = torch.zeros((args.buffer_size, 1, 1),dtype=float64)# (v/args.eps).log1p() if choose else v*args.cof
    # Feature of UEs
    data['UE'].x = torch.cat((clean_p_AP, clean_p_UAV),dim=-2).transpose(-2, -1)
    data['UE'].pilot_onehot  = pilot_onehot
    data['UE'].receive_noise = receive_noise.unsqueeze(-1)

    data['UE'].x = data['UE'].x.to(float64)
    data['UE'].pilot_onehot  = data['UE'].pilot_onehot .to(float64)
    data['UE'].receive_noise = data['UE'].receive_noise.to(float64)

    data['UE'].power = data['UE'].x # 最优解
    data['AP'].v     = data['AP'].x # 最优解

    src1 = torch.arange(args.ap_num+args.uav_num,device=args.device).repeat_interleave(args.ue_num)
    dst1 = torch.arange(args.ue_num,device=args.device).repeat(args.ap_num+args.uav_num)
    edge_ap_ue = torch.stack([src1, dst1], dim=0).unsqueeze(0).repeat((args.buffer_size, 1, 1)).to(args.device)
    edge_ue_ap = edge_ap_ue.clone().flip(-2)
    data['AP', 'connect', 'UE'    ] .edge_index = edge_ap_ue
    data['UE', 'rev_connect', 'AP'] .edge_index = edge_ue_ap
    
    src2 = torch.arange(1,device=args.device).repeat_interleave(args.ap_num+args.uav_num)
    dst2 = torch.arange(args.ap_num+args.uav_num,device=args.device).repeat(1)
    edge_cpu_ap = torch.stack([src2, dst2], dim=0).unsqueeze(0).repeat((args.buffer_size, 1, 1)).to(args.device)
    edge_ap_cpu = edge_cpu_ap.clone().flip(-2)
    data['CPU', 'connect', 'AP'    ] .edge_index = edge_cpu_ap
    data['AP', 'rev_connect', 'CPU'] .edge_index = edge_ap_cpu


    ap_indices_ap_ue = edge_ap_ue[:,-2]
    ue_indices_ap_ue = edge_ap_ue[:,-1]
    edge_features_ap_ue = beta[torch.arange(args.buffer_size).unsqueeze(1), ap_indices_ap_ue, ue_indices_ap_ue].unsqueeze(-1)

    edge_features_ue_ap = edge_features_ap_ue.clone()

    data['AP', 'connect'    , 'UE'].edge_attr = (edge_features_ap_ue/args.eps).log1p() if choose else edge_features_ap_ue
    data['UE', 'rev_connect', 'AP'].edge_attr = (edge_features_ue_ap/args.eps).log1p() if choose else edge_features_ue_ap
    data['AP', 'connect'    , 'UE'].edge_attr, data['UE', 'rev_connect', 'AP'].edge_attr = \
    data['AP', 'connect'    , 'UE'].edge_attr.to(float64), data['UE', 'rev_connect', 'AP'].edge_attr.to(float64)

    edge_features_cpu_ap = C.unsqueeze(-1) 
    edge_features_ap_cpu = edge_features_cpu_ap.clone()
    data['CPU', 'connect'    , 'AP' ].edge_attr = (edge_features_cpu_ap/args.eps).log1p() if choose else edge_features_cpu_ap
    data['AP' , 'rev_connect', 'CPU'].edge_attr = (edge_features_ap_cpu/args.eps).log1p() if choose else edge_features_ap_cpu

    data['CPU', 'connect'    , 'AP' ].edge_attr, data['AP' , 'rev_connect', 'CPU'].edge_attr = \
    data['CPU', 'connect'    , 'AP' ].edge_attr.to(float64), data['AP' , 'rev_connect', 'CPU'].edge_attr.to(float64)   
    return data


if __name__ == '__main__':
    args = get_args()
    dataset = torch.load(r'dataset\1920_dataset.pt', weights_only=False)
    model_path = None
    data_create(args, dataset)