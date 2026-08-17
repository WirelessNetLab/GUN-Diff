import matplotlib.pyplot as plt

def device_show(location_index:dict, block):

    plt.rcParams['axes.unicode_minus'] = False           # 正常显示负号 
    plt.figure(figsize=(8, 6))
    
    ap_index  = location_index ['ap' ]
    uav_index = location_index ['uav']
    ue_index  = location_index ['ue' ]
    cpu_index = location_index ['cpu']

    plt.scatter(cpu_index [:,0], cpu_index [:,1], c='red'    ,  marker='*', s=512, label='CPU' )
    plt.scatter(ap_index  [:,0], ap_index  [:,1], c='blue'   ,  marker='^', s=64, label='AP'  )
    plt.scatter(uav_index [:,0], uav_index [:,1], c='orange' ,  marker='v', s=64,  label='UAV' )
    plt.scatter(ue_index  [:,0], ue_index  [:,1], c='black'  ,  marker='o', s=16,  label='UE'  )
    




    plt.xlabel("X ")
    plt.ylabel("Y ")
    # plt.title("")
    plt.grid(True)
    plt.xlim(-block//10, block + block//10)
    plt.ylim(-block//10, block + block//10)
    plt.legend()
    plt.axis('equal')  # 保持比例

    plt.plot([0     , block], [0    , 0     ], color='black', linestyle='-', linewidth=3)  # down
    plt.plot([0     , block], [block, block ], color='black', linestyle='-', linewidth=3)  # up
    plt.plot([0     , 0    ], [0    , block ], color='black', linestyle='-', linewidth=3)  # left
    plt.plot([block , block], [0    , block ], color='black', linestyle='-', linewidth=3)  # right



    plt.show()