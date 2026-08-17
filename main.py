from get_args import get_args
from my_function import Diffusion
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Deep Learning
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from torch import nn, optim
from my_function.deep_learning.Network import GraphUNet
from data_create import data_create
from utils import correct_layer, get_DS,compute_IN_vals_vec, MyLrScheduler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#%% Train Model

def disable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0

def main(args, dataset):
    torch.set_default_dtype(torch.float64) 
    data = data_create(args=args,dataset=dataset)
    train_size = int(args.buffer_size * args.training_percent)
    valid_size=args.buffer_size - train_size

    model = GraphUNet(  dim_cpu = 1,
                        dim_ap  = 1,
                        dim_ue  = args.ap_num+args.uav_num,
                        hidden_dim = args.nhid,
                        dropout = args.dropout,
                        ).to(args.device).to(torch.float64)

    Diff_tool = Diffusion(model=model,
                          beta_min=args.beta_min, 
                          beta_max=args.beta_max,
                          T=args.T,
                          beta_mode=args.beta_mode
                        ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = MyLrScheduler(
                                    optimizer,
                                    schedule=[
                                        0, (2e-4, 200),
                                        0, (1e-4, 50 ),
                                        0, (5e-5, 50 ),
                                        0, (1e-5, 50 ),
                                        0, (5e-6, 50 ),
                                        0, (5e-6, 1e-5, 20),
                                    ],
                                    base_lr=0.0
                                )
    loss_func = nn.MSELoss()
    AP_train_save, UE_train_save, AP_valid_save, UE_valid_save = [], [], [], []
    loss_train_save = [3]
    loss_valid_save = [3]
    SE_save = []
    SE_valid_save = []
    
    E_freeze = 20
    for epoch in range(1, args.epochs):
        disturb_idx = torch.randperm(train_size)
        if epoch == E_freeze:
            model.apply(disable_dropout)
            print(r'Dropout Finished')

        start_time   = datetime.now()
        data_input   = data.clone()
        t_dif        = torch.randint(1,args.T+1, size=(args.buffer_size,1,),device=args.device)
        data_input.t = t_dif
        data_input['AP' ].noise = torch.zeros_like(data_input['AP' ].x,)
        data_input['AP' ].x[:,-args.uav_num:]    , _ , data_input['AP' ].noise[:,-args.uav_num:] = Diff_tool.Forward_Diffusion  (data['AP' ].x[:,-args.uav_num:] ,t = t_dif)
        data_input['UE' ].x    , _ , data_input['UE' ].noise       = Diff_tool.Forward_Diffusion  (data['UE' ].x ,t = t_dif)
        for key, tensor in data_input.x_dict.items():
            if torch.isnan(tensor).any():
                raise KeyError(f"Key '{key}' has NaN.")

        data_input      = split_batched_heterodata(data_input)
        train_dataset   = [data_input[i] for i in disturb_idx.tolist()]
        valid_dataset   = data_input[-valid_size: ]
        train_loader    = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False )
        valid_loader    = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False )

        train_loss  ,  train_loss_history, SE_history       = train   (args, Diff_tool, model, train_loader, optimizer, \
                                                                 loss_func, args.device, disturb_idx,dataset, epoch)
        scheduler.step()
        valid_loss  ,  valid_loss_history, SE_history_valid = valid(args, Diff_tool, model, valid_loader, \
                                                                 loss_func, args.device, dataset)
        AP_train, UE_train, AP_valid, UE_valid = train_loss_history['AP'], train_loss_history['UE'], valid_loss_history['AP'], valid_loss_history['UE']

        AP_train_save   .append(AP_train), UE_train_save.append(UE_train)
        AP_valid_save   .append(AP_valid), UE_valid_save.append(UE_valid)
        SE_save         .append(SE_history      )
        SE_valid_save   .append(SE_history_valid)

        end_time = datetime.now()
        print(f"│ Epoch {epoch} │ {end_time} │ Use {((end_time-start_time).seconds % 60)}s │ {train_loss:.4f} {'↘' if train_loss-loss_train_save[-1]<=0 else '↗'} │ Valid: {valid_loss:.4f} {'↘' if valid_loss-loss_valid_save[-1]<=0 else '↗'} │ LR:{optimizer.param_groups[0]['lr']:.2e} │ {"√" if train_loss>valid_loss else "×"}\n")
        loss_train_save.append( train_loss )
        loss_valid_save.append( valid_loss )
    ################################################################################################
    now = datetime.now()
    os.makedirs (os.path.join(args.data_logdir , args.Net_structure, f'{now.strftime("%Y_%m_%d_%H_%M")}'), exist_ok=True)
    data_path  = os.path.join(args.data_logdir , args.Net_structure, f'{now.strftime("%Y_%m_%d_%H_%M")}',  f'{now.strftime("detailed_data_%Y_%m_%d_%H_%M")}.csv' )
    
    os.makedirs (os.path.join(args.model_logdir , args.Net_structure, f'{now.strftime("%Y_%m_%d_%H_%M")}'), exist_ok=True)
    model_path = os.path.join(args.model_logdir , args.Net_structure, f'{now.strftime("%Y_%m_%d_%H_%M")}', f'{loss_train_save[-1]:.4f}_{now.strftime("%Y_%m_%d_%H_%M")}.pth'       )
    args_path  = os.path.join(args.model_logdir , args.Net_structure, f'{now.strftime("%Y_%m_%d_%H_%M")}', f'args.pth'       )
    
    # Save Model ################################################################################################
    torch.save(model.state_dict(),model_path )

    torch.save(args, args_path  )
    print('Model & Args Saved finished')

    # Loss ################################################################################################
    df = pd.DataFrame({
                            "MSE_loss"          : loss_train_save,     # Train Loss
                            "MSE_loss_test"     : loss_valid_save,     # Valid Loss
                            "train_SE" : SE_save,
                            "valid_SE" : SE_valid_save,
                        })
    
    df.to_csv(data_path, index=False)
    
    print('Data save Finished')


def split_batched_heterodata(data: HeteroData) -> list[HeteroData]:
    """
    split batched HeteroData to HeteroData in single batch.
    """
    num_samples = data['AP'].x.size(0)
    data_list = []

    for i in range(num_samples):
        d = HeteroData()

        for key, value in data.items():
            if isinstance(value, (str, int, float)):
                d[key] = value
            elif hasattr(value, '__getitem__') and value.size(0) == num_samples:
                d[key] = value[i]
            else:
                d[key] = value

        # Vertex
        for node_type in data.node_types:
            for key, value in data[node_type].items():
                if isinstance(value, (str, int, float)):
                    d[node_type][key] = value
                elif hasattr(value, '__getitem__') and value.size(0) == num_samples:
                    d[node_type][key] = value[i]
                else:
                    d[node_type][key] = value

        # Edge
        for edge_type in data.edge_types:
            for key, value in data[edge_type].items():
                if isinstance(value, (str, int, float)):
                    d[edge_type][key] = value
                elif hasattr(value, '__getitem__') and value.size(0) == num_samples:
                    d[edge_type][key] = value[i]
                else:
                    d[edge_type][key] = value

        data_list.append(d)

    return data_list


def monitor_gradients(model, prefix="ue", print_detail:bool=False):
    """
    Monitor the gradients of prefix
    Args:
        model: your torch model
        prefix: keyword of the monitored part
        print_detail: print in each epoch?
    """
    grads = []

    for name, p in model.named_parameters():
        if prefix in name and p.grad is not None:
            grad_norm = p.grad.detach().norm(2)
            grads.append(grad_norm)

            if print_detail:
                print(f"[{prefix}] {name:40s} | grad_norm = {grad_norm:.4e}")

    if len(grads) == 0:
        print(f"[{prefix}] No gradients found.")
        return

    grads = torch.stack(grads)

    print(
        f"[{prefix} Grad Summary] "
        f"mean = {grads.mean():.4e}, "
        f"max = {grads.max():.4e}, "
        f"min = {grads.min():.4e}"
    )


def train(args, Diff_tool:Diffusion, model:GraphUNet, loader, optimizer, loss_func, device, disturb_idx, dataset, epoch):
    """
    train function
    """
    # assert isinstance(Diff_tool, Diffusion)
    model.train()
    total_loss       = 0
    power_total_loss = 0
    v_total_loss     = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        power_loss  = loss_func( out ['UE'] .x, batch['UE'] .noise)
        v_loss      = loss_func( out ['AP'] .x    .reshape(args.batch_size, -1, 1)[:, -args.uav_num:,] , \
                                batch['AP'] .noise.reshape(args.batch_size, -1, 1)[:, -args.uav_num:,]    )

        loss = power_loss + v_loss
        loss .backward()
        clip_grad_norm_(model.parameters(), 1)

        optimizer.step()
        total_loss       += loss        .item()
        power_total_loss += power_loss  .item()
        v_total_loss     += v_loss      .item()



    if Diff_tool is not None:
    #  ######################################################################################################
        T = 8
        with torch.no_grad():
            result = Diff_tool.DDIM(args, batch, T)
        power_temp, v_temp = result['UE'].x.reshape(args.batch_size, -1,result['UE'].x.shape[-1]).transpose(-1, -2).squeeze(1), result['AP'].x.reshape(args.batch_size, -1, 1)[:, -args.uav_num:, :].transpose(-1, -2)
        power_no_correct, v_no_correct      = power_temp, v_temp

        power, v = correct_layer(args, power_no_correct, v_no_correct)


        v_m = np.ones((args.batch_size, args.ap_num,))/args.ap_num
        # Equal wireless fronthaul resource in each AAP q for UEs ###############################
        rho_qk   = np.ones((args.batch_size, args.uav_num, args.ue_num)) /args.ue_num
        rho_mk   = np.ones((args.batch_size, args.ap_num , args.ue_num)) /args.ue_num

        IN = compute_IN_vals_vec(args,
                    power[:, :args.ap_num].numpy(), power[:, args.ap_num:].numpy(),              # (M,K), (Q,K)
                    rho_mk, rho_qk,             # (M,K), (Q,K)
                    v_m, v.squeeze(0).numpy(),  # (M, ), (Q, )
                    dataset, 
                    disturb_idx[-args.batch_size:,]
                    )
        DS2 = get_DS(args, power, dataset, disturb_idx[-args.batch_size:,])

        IN_no_correct   = compute_IN_vals_vec(args,
                                                power_no_correct.reshape(args.batch_size, -1, args.ue_num)[:, :args.ap_num].numpy(), power_no_correct.reshape(args.batch_size, -1, args.ue_num)[:, args.ap_num:].numpy(),              # (M,K), (Q,K)
                                                rho_mk, rho_qk,                         # (M,K), (Q,K)
                                                v_m, v_no_correct.squeeze(-2).numpy(),  # (M, ), (Q, )
                                                dataset, 
                                                disturb_idx[-args.batch_size:,]
                                                )
        DS2_no_correct = get_DS(args, power_no_correct.reshape(args.batch_size, -1, args.ue_num), dataset, disturb_idx[-args.batch_size:,])

        current_SE    = (args.tau_d/args.tau_c)*np.sum(np.log2(1+DS2/IN),axis=-1)
        no_correct_SE = (args.tau_d/args.tau_c)*np.sum(np.log2(1+DS2_no_correct/IN_no_correct),axis=-1)
        SE_expert = dataset['SE_expert'][disturb_idx[-args.batch_size:,]].numpy()

    # ######################################################################################################
  
    loss_history = {
                    'UE'        :power_total_loss / len(loader) ,
                    'AP'        :v_total_loss     / len(loader) ,
                    }
    SE_history = None
    if Diff_tool is not None:
        SE_history = {
                        "SE_pred":current_SE,
                        "no_correct_SE":no_correct_SE,
                        "SE_expert":SE_expert,
                    }
    # 
    # monitor_gradients(model)
    # monitor_gradients(model,prefix="ap")
    return total_loss / len(loader), loss_history, current_SE.mean()

@torch.no_grad()
def valid(args, Diff_tool, model, loader, loss_func, device, dataset):
    """
    validation function
    """
    total_loss       = 0
    power_total_loss = 0
    v_total_loss     = 0
    for batch in loader:
        batch = batch.to(device)

        out = model(batch)
        power_loss  = loss_func( out ['UE'] .x, batch['UE'] .noise       )
        v_loss      = loss_func( out ['AP'] .x    .reshape(args.batch_size, -1, 1)[:, -args.uav_num:,] , \
                                batch['AP'] .noise.reshape(args.batch_size, -1, 1)[:, -args.uav_num:,]    )

        loss = power_loss + v_loss
        total_loss          += loss         .item()
        power_total_loss    += power_loss   .item()
        v_total_loss        += v_loss       .item()
    if Diff_tool is not None:
    #  ######################################################################################################
        T = 8
        idx = torch.arange(args.buffer_size*args.training_percent+args.batch_size, args.buffer_size, dtype=torch.int)
        with torch.no_grad():
            result = Diff_tool.DDIM(args, batch, T)
        power_temp, v_temp = result['UE'].x.reshape(args.batch_size, -1,result['UE'].x.shape[-1]).transpose(-1, -2).squeeze(1), result['AP'].x.reshape(args.batch_size, -1, 1)[:, -args.uav_num:, :].transpose(-1, -2)
        # power_no_correct, v_no_correct = result['UE'].x.reshape(args.batch_size, -1,result['UE'].x.shape[-1]).transpose(-1, -2).squeeze(1), result['AP'].x.reshape(args.batch_size, -1, 1)[:, -args.uav_num:, :].transpose(-1, -2)
        power_no_correct, v_no_correct = power_temp, v_temp
        power, v = correct_layer(args, power_no_correct, v_no_correct)

        v_m = np.ones((args.batch_size, args.ap_num,))/args.ap_num
        # Equal wireless fronthaul resource in each AAP q for UEs ###############################
        rho_qk   = np.ones((args.batch_size, args.uav_num, args.ue_num)) /args.ue_num
        rho_mk   = np.ones((args.batch_size, args.ap_num , args.ue_num)) /args.ue_num

        IN = compute_IN_vals_vec(args,
                    power[:, :args.ap_num].numpy(), power[:, args.ap_num:].numpy(), # (M,K), (Q,K)
                    rho_mk, rho_qk, # (M,K), (Q,K)
                    v_m, v.numpy(), # (M, ), (Q, )
                    dataset, 
                    idx,
                    )
        DS2 = get_DS(args, power, dataset, idx)

        IN_no_correct   = compute_IN_vals_vec(args,
                                                power_no_correct.reshape(args.batch_size, -1, args.ue_num)[:, :args.ap_num].numpy(), power_no_correct.reshape(args.batch_size, -1, args.ue_num)[:, args.ap_num:].numpy(),              # (M,K), (Q,K)
                                                rho_mk, rho_qk,                         # (M,K), (Q,K)
                                                v_m, v_no_correct.squeeze(-2).numpy(),  # (M, ), (Q, )
                                                dataset, 
                                                idx
                                                )
        DS2_no_correct = get_DS(args, power_no_correct.reshape(args.batch_size, -1, args.ue_num), dataset, idx)

        current_SE    = (args.tau_d/args.tau_c)*np.sum(np.log2(1+DS2/IN),axis=-1)
        no_correct_SE = (args.tau_d/args.tau_c)*np.sum(np.log2(1+DS2_no_correct/IN_no_correct),axis=-1)
        SE_expert = dataset['SE_expert'][idx].numpy()
    #  ######################################################################################################

    loss_history = {
                    'UE'        :power_total_loss / len(loader),
                    'AP'        :v_total_loss     / len(loader),
                    }
    SE_history = None
    if Diff_tool is not None:
        SE_history = {
                        "SE_pred"       :current_SE,
                        "no_correct_SE" :no_correct_SE,
                        "SE_expert"     :SE_expert,
                    }
    return total_loss / len(loader), loss_history, current_SE.mean()

if __name__ == '__main__':
    args = get_args()
    dataset = torch.load(r'dataset\1920_dataset.pt', weights_only=False)
    
    main(args, dataset) # 训练主函数
