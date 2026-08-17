import torch
def scale_input(x, eps=1e-7):
    # x = x + eps                           # Bias for avoiding 0
    x = torch.log1p(x / eps)                
    # x = (x - x.min()) / (x.max() - x.min()) 
    return x

def inverse_scale_input(x_scaled, eps=1e-7):# , original_min, original_max, ):
    """
    Ags:
        x_scaled : inverse scaled input
        eps : Bias for avoiding 0

    
    Return:
        x_recovered : recovered data

    """
    # x_log = x_scaled * (original_max - original_min) + original_min
    
    # x_recovered = (torch.expm1(x_log)) * eps
    
    # return x_recovered
    x_recovered = (torch.exp(x_scaled) - 2) * eps
    return x_recovered