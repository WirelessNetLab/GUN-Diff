import numpy as np
import matplotlib.pyplot as plt





def location_generate(args, test):
    """
    Args:
        args : including Block Length, Devide Number, etc.

    Returns:
        ap_index  : AP  location index (ap_num,  ap_x,  ap_y  )
        uav_index : UAV location index (uav_num, uav_x, uav_y )
        ue_index  : UE  location index (ue_num,  ue_x,  ue_y  )
        cpu_index : CPU location index (1,       cpu_x, cpu_y )
    """
    if test:
        np.random.seed(42)
    if args.choice == 'random':
        # 1. Random Location generate
        ap_index  = np.random.randint(0, args.block_lenth + 1,(args.ap_num,2)  )
        uav_index = np.random.randint(0, args.block_lenth + 1,(args.uav_num,2) )
        ue_index  = np.random.randint(0, args.block_lenth + 1,(args.ue_num,2)  )
        cpu_index = np.array([args.block_lenth/2, args.block_lenth/2])[np.newaxis,:]


    elif args.choice == 'uniform':
        # 2. uniform distribution
        ap_block_num = args.ap_num
        uav_block_num = args.uav_num
        
        ue_index  = np.random.randint(0, args.block_lenth + 1,(args.ue_num,2)  )
        cpu_index = np.array([args.block_lenth/2, args.block_lenth/2])[np.newaxis,:]

        ap_grid_size = int(np.ceil(np.sqrt(args.ap_num)))
        cell_size = args.block_lenth / ap_grid_size
        lin_coords = np.linspace(cell_size / 2, args.block_lenth - cell_size / 2, ap_grid_size)
        xx, yy = np.meshgrid(lin_coords, lin_coords)
        ap_index = np.stack([xx.ravel(), yy.ravel()], axis=1)

        uav_grid_size = int(np.ceil(np.sqrt(args.uav_num)))
        cell_size = args.block_lenth / uav_grid_size
        lin_coords = np.linspace(cell_size / 2, args.block_lenth - cell_size / 2, uav_grid_size)
        xx, yy = np.meshgrid(lin_coords, lin_coords)
        uav_index = np.stack([xx.ravel(), yy.ravel()], axis=1)
        

    locationa_index = {}
    locationa_index [ 'ap'  ]       = np.concatenate( (ap_index,  np.ones ((args.ap_num, 1,))* args.h_ap      ), axis=1)
    locationa_index [ 'uav' ]       = np.concatenate( (uav_index, np.ones ((args.uav_num,1,))* args.uav_level ), axis=1)
    locationa_index [ 'ue'  ]       = np.concatenate( (ue_index,  np.ones ((args.ue_num, 1,))* args.h_u       ), axis=1)
    locationa_index [ 'cpu' ]       = np.concatenate( (cpu_index, np.zeros((1          , 1,))                 ), axis=1)

    return locationa_index
