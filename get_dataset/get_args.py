
def get_args():
    """
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type = str, default = "cpu")

    "————————————————————————————————— |coef of communication| —————————————————————————————————"
    ############################|  Communication Device |##########################
    parser.add_argument("--cpu-num", type = int,   default = 1    )
    parser.add_argument("--uav-num", type = int,   default = 8    )
    parser.add_argument("--ap-num" , type = int,   default = 12   )
    parser.add_argument("--ue-num" , type = int,   default = 10   )

    # Antenna Num
    parser.add_argument("--cpu-antenna",  type = int,   default = 8   )
    parser.add_argument("--ap-antenna" ,  type = int,   default = 4   )
    parser.add_argument("--uav-antenna",  type = int,   default = 4   )
    parser.add_argument("--ue-antenna" ,  type = int,   default = 1   )

    ##############################################################

    parser.add_argument("--block-lenth",   type = float,   default = 1000      )
    parser.add_argument("--choice"     ,   type = str,   default = 'random', 
                                                  choices=['uniform', 'random']  )

    # Band
    parser.add_argument("--B-A",               type = float,   default = 180e-3     ) # MHz # Access Bandwidth
    parser.add_argument("--B-F",               type = float,   default = 10         ) # MHz # Fronthaul Bandwidth

    # Height
    parser.add_argument("--uav-level",         type = float,   default = 100        ) # UAV 
    parser.add_argument("--h-ap",              type = float,   default = 15         ) # AP  
    parser.add_argument("--h-u",               type = float,   default = 1.65       ) # UE  


    # Coef of Limited Capacity 
    parser.add_argument("--wired-capacity",    type = float,   default = 20          ) # bps/Hz 
    parser.add_argument("--power-cpu-in-dBm",  type = float,   default = 47          ) # dBm
    parser.add_argument("--N0",                type = float,   default = 4e-21       ) # 4e-21


    # Carrier
    parser.add_argument("-f",                  type = float,   default = 1.9e3   ) # MHz
    parser.add_argument("--f-F",               type = float,   default = 5e3     ) # MHz

    # Power upper bound of APs
    parser.add_argument("--Pm-ap",             type = float,   default = 30    ) # dBm
    parser.add_argument("--Pm-uav",            type = float,   default = 30    ) # dBm

    parser.add_argument("--xi-1",                type = float,   default = 11.95     ) # 
    parser.add_argument("--xi-2",                type = float,   default = 0.136     ) # 

    # Time-Freq Resource Block
    parser.add_argument("--tau-c",             type = int,   default = 200     ) #
    parser.add_argument("--tau-p",             type = int,   default = 10      ) #
    # Precoed method
    parser.add_argument("--method",            type = str,   default = 'Conj', 
                                                       choices=['ZF', 'Conj']  )

    parser.add_argument("--solver",            type = str,   default = 'DCP', 
                                                    choices=['WMMSE', 'DCP']          )
    parser.add_argument("--max-iter",          type = int,   default = 30)
    parser.add_argument("--alpha",             type = float,   default = 0.3)

    # Diffusion
    parser.add_argument("-T",                     type = int  ,   default = 1000      ) # Total Timesteps
    parser.add_argument("--beta-min",             type = float,   default = 1e-4      ) # min beta
    parser.add_argument("--beta-max",             type = float,   default = 2e-2      ) # max beta
    parser.add_argument("--beta-mode",            type = str  ,   default = 'lin',
                                                   choices=['lin', 'vp', 'cosine']     )
    parser.add_argument("--eps",                  type = float,   default = 1e-9      ) 
    
    # GNN
    parser.add_argument("--Net-structure",         type = str  ,     default = 'HeteroGUNet',
                                                   choices=['GAT','GCN','Graphormer', 'GATUNet', 'HeteroGUNet'] )

    parser.add_argument("--act",                   type = str   ,    default = 'SiLU',
                                                        choices=['SiLU','ReLU','Mish'])
    parser.add_argument("--aggr",                  type = str   ,    default = 'sum',
                                                        choices=['sum', 'mean','max','min'])
    
    parser.add_argument("--layer-norm",            type = bool   ,     default = True    )   # Layer_norm
    parser.add_argument("--nhid",                  type = int    ,     default = 128     )   # Hidden Dim
    parser.add_argument("--buffer-size",           type = int    ,     default = 640     )   # Buffer Size
    parser.add_argument("--batch-size",            type = int    ,     default = 64      )   # Batch Size
    parser.add_argument("--cof",                   type = int    ,     default = 1       )
    parser.add_argument("--training-percent",      type = float  ,     default = 0.8     )
    parser.add_argument("--lr",                    type = float  ,     default = 1e-4    )   # Learning Rate
    parser.add_argument("--pool-ratio",            type = float  ,     default = 1/2     )   # Pool Ratio
    parser.add_argument("--dropout",               type = float  ,     default = 0.00    )   # Dropout
    parser.add_argument("--epochs",                type = int    ,     default = 100_000 )   # 
    parser.add_argument("--get-noise",             type = bool   ,     default = True    )   

    parser.add_argument("--model-logdir",          type = str    ,     default = 'model' ) # model save dir
    parser.add_argument("--data-logdir",           type = str    ,     default = 'data'  ) # data save dir

    # switch
    parser.add_argument("--device-show",           type = bool   ,   default = False    ) # show device


    args = parser.parse_known_args()[0]
    args.vald_percent = 1. - args.training_percent
    args.tau_d        = args.tau_c - args.tau_p

    return args
