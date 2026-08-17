import numpy as np

def pilot_generate(args:int):
    """
    pilot generate function

    Args:
        pilot_lenth (int) : pilot length

    Returns:
        pilots (ndarray): pilot sequences, shape = (tau_p, tau_p)
    """
    pilot_lenth = args.tau_p
    assert isinstance(pilot_lenth,int), "The pilot length must be int"
    n = np.arange(pilot_lenth)
    k = n.reshape((pilot_lenth, 1))
    temp = np.exp(-2j * np.pi * k * n / pilot_lenth) / np.sqrt(pilot_lenth)
    pilots = temp / np.linalg.norm(temp, axis=1, keepdims=True)
    return pilots

def pilot_allocate(pilots, ue_num:int, test):
    """
    Args:
        pilots : pilot sequences, shape=(pilot_num, pilot_lenth)
        ue_num : Number of UEs

    Returns:
        tuple :
             - pilot_index : Pilot index
             - ue_pilot    : UE Pilot sequences
    """
    if test:
        np.random.seed(42)
    pilot_num = pilots.shape[0]

    if (ue_num - pilot_num) <= 0:
        pilot_index = np.arange(ue_num)
    else:
        pilot_index = np.random.permutation(np.array([i % pilot_num for i in range(ue_num)]))
    ue_pilot = pilots[pilot_index,:]
    return pilot_index, ue_pilot
