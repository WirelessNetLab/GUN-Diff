import torch
from torch import nn
import numpy as np
from .beta_schedule import get_beta_schedule
from .utils import scale_input
import time
from torch_geometric.data import HeteroData
from torch import float32, sqrt, cumprod, clamp, log, float64

class Diffusion(nn.Module):
    def __init__(self, model, beta_min, beta_max, T, beta_mode):
        super(Diffusion, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_mode = beta_mode
        self.T = T
        self.model = model
        self.sigmoid = nn.Sigmoid()

        self.beta = get_beta_schedule(mode = self.beta_mode, beta_min = self.beta_min, beta_max = self.beta_max, timesteps = self.T)
        self.alpha = 1. - self.beta
        self.alphas_cumprod = cumprod(self.alpha, axis=0)
        self.alphas_cumprod_prev = torch.tensor(np.append(1., self.alphas_cumprod[:-1]))
        self.sqrt_alphas_cumprod = sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 =self.beta * sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(self.alpha) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.beta * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.beta_ = torch.zeros_like(self.beta)
        self.posterior_log_variance_clipped = log(clamp(self.posterior_variance, min=1e-20))

        for i in range(len(self.beta)):
            if i == 0:
                self.beta_[i] = self.beta[i]
            else:
                self.beta_[i] = (self.alpha[i-1]/self.alpha[i]) * self.beta[i]
        time.sleep(0)
        temp = torch.arange(1, self.T,dtype=torch.int)
        self.steplist = self.T/temp
        mask = (self.steplist%1)==0
        self.steplist = temp[mask]


    def Forward_Diffusion(self, input, t:int, eps=None):

        assert isinstance(input, torch.Tensor)
        if eps is not None:
            input = scale_input(input, eps=eps)
        noise = torch.normal(mean = 0, std = 1, size = input.shape,device=input.device, dtype=float64)
        noisy_input = self.sqrt_alphas_cumprod[t-1].unsqueeze(-1) * input +\
            self.sqrt_one_minus_alphas_cumprod[t-1].unsqueeze(-1) * noise
        data_taget = self.sqrt_alphas_cumprod[t-2].unsqueeze(-1) * input +\
            self.sqrt_one_minus_alphas_cumprod[t-2].unsqueeze(-1) * noise
        return noisy_input, data_taget, noise

    def DDIM(self, args, data_orig:HeteroData, T):
        """
        Args:
            data_orig : original data
            T         : Sampling steps, with a step size of 1000/T
        """
        if (self.T/T)%1 != 0.:
            raise ValueError(f"The time step is not an integer, and the total number of diffusion steps is: {self. T}, which needs to be divided evenly")
        step = self.T//T
        data = data_orig.clone()
        data['UE'].power, data['AP'].v = data['UE'].x, data['AP'].x
        data['UE'].x, data['AP'].x = torch.randn_like(data['UE'].x), torch.randn_like(data['AP'].x)
        with torch.no_grad():
            for t in reversed(range(step, self.T+1, step)):
                data.t = t*torch.ones((data.num_graphs, ),dtype=data['AP'].x.dtype)
                data = self.model(data)
                xt =    {
                            "AP" :data['AP'].xt,
                            "UE" :data['UE'].xt,
                        }
                noise_pred = {
                                "AP":data['AP'].noise_pred,
                                "UE":data['UE'].noise_pred,
                            }
                coef = self.alphas_cumprod[t-step-1] if t-step !=0 else torch.ones((1,))
                eta = 0.0
                sigma = eta * sqrt( (1-coef) / (1-self.alphas_cumprod[t-1]) ) * sqrt(1-(self.alphas_cumprod[t-1]/coef))
                coef1 = sqrt(coef/self.alphas_cumprod[t-1])
                coef2 = sqrt(coef)
                coef3 = sqrt((1-coef-sigma**2)/coef)
                coef4 = sqrt((1-self.alphas_cumprod[t-1])/self.alphas_cumprod[t-1])
                if t == step:
                    None
                x_pred = {
                    "AP": (coef1*xt["AP"] + coef2*(coef3-coef4)*noise_pred["AP"] + sigma**2*torch.randn_like(noise_pred["AP"])),
                    "UE": (coef1*xt["UE"] + coef2*(coef3-coef4)*noise_pred["UE"] + sigma**2*torch.randn_like(noise_pred["UE"])),
                }
                if t <= self.T/2:
                    x_pred["AP"].clamp_(min=-1, max=1)
                    x_pred["UE"].clamp_(min=-1, max=1)
                data['UE'].x, data['AP'].x =x_pred['UE'], x_pred['AP']
                print(f"From Step: {t} to {t-step}\t|",end='\t'if t != step else'\n')
        print(f" Sample Finished")
        power_pred, v_pred = data['UE'].x, data['AP'].x
        power_expert, v_expert = data['UE'].power, data['AP'].v
        return data

    def DDPM(self, data, timesteps, *args, **kwargs): 
        self.idx = torch.arange(0, timesteps)
        result = self.p_sample_loop(data, timesteps, *args, **kwargs)
        return result
    
    def p_sample_loop(self, data:HeteroData, timesteps):

        for timestep in reversed(range(1, timesteps+1)):

            data = self.p_sample(data, timestep)
            print(timestep, end=' ' if timestep != 1 else '\n')
            data.t -= 1

        power, v = data['AP'].x, data['CPU'].x
        data['AP'].x, data['CPU'].x = power, v
        return data 
    
    def p_sample(self, data:HeteroData, timestep):
        result = data.clone()
        model_mean, _, model_log_variance = self.p_mean_variance(data=data, t=timestep,)

        noise = {"AP" :torch.randn_like(data['AP'].x),
                 "CPU":torch.randn_like(data['CPU'].x),}
        
        power_t = model_mean['AP' ] + (0.5 * model_log_variance['AP' ]).exp() * noise['AP' ]
        v_t     = model_mean['CPU'] + (0.5 * model_log_variance['CPU']).exp() * noise['CPU']
        power_t_result, v_t_result = power_t, v_t      
        
        result['AP'].x, result['CPU'].x = power_t_result, v_t_result
        return result
    
    def p_mean_variance(self, data, t):
        dataout = self.model(data)
        x_recon = self.predict_start_from_noise(data, t, dataout)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, data=dataout, t=t,)
        return model_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(self, data_t, t, dataout):

        noise_pred = {  "CPU" : dataout['CPU'].x,
                        "AP"  : dataout['AP' ].x,
                     }
        xt = {  "CPU" : dataout['CPU'].xt,
                "AP"  : dataout['AP' ].xt,
             }
        x_recon = {"CPU":(extract(self.sqrt_recip_alphas_cumprod  , self.idx[(dataout.t-1).to(int)],  xt['CPU'].shape) * xt['CPU'] - \
                          extract(self.sqrt_recipm1_alphas_cumprod, self.idx[(dataout.t-1).to(int)],  xt['CPU'].shape) * noise_pred['CPU']),
                  "AP" :(extract(self.sqrt_recip_alphas_cumprod   , self.idx[(dataout.t-1).to(int)],  xt['AP'].shape) * xt['AP' ] - \
                         extract(self.sqrt_recipm1_alphas_cumprod , self.idx[(dataout.t-1).to(int)],  xt['AP'].shape) * noise_pred['AP' ]),
                  }
        return x_recon


    def q_posterior(self, x_start, data, t):
        xt = {  "CPU" : data['CPU'].xt,
                "AP"  : data['AP' ].xt,
             }
        posterior_mean_ap = (
                extract(self.posterior_mean_coef1, self.idx[(data.t-1).to(int)], xt['AP'].shape) * x_start['AP'] +
                extract(self.posterior_mean_coef2, self.idx[(data.t-1).to(int)], xt['AP'].shape) * xt      ['AP']
        )
        posterior_mean_cpu = (
                extract(self.posterior_mean_coef1, self.idx[(data.t-1).to(int)], xt['CPU'].shape) * x_start['CPU'] +
                extract(self.posterior_mean_coef2, self.idx[(data.t-1).to(int)], xt['CPU'].shape) * xt     ['CPU']
        )
        posterior_mean = {"CPU": posterior_mean_cpu.to(float32),
                          "AP" : posterior_mean_ap .to(float32),
                          }
        posterior_variance_ap  = extract(self.posterior_variance, self.idx[(data.t-1).to(int)], xt['AP' ].shape)
        posterior_variance_cpu = extract(self.posterior_variance, self.idx[(data.t-1).to(int)], xt['CPU'].shape)
        posterior_variance = {  "CPU": posterior_variance_cpu.to(float32),
                                "AP" : posterior_variance_ap .to(float32),
                             }
        posterior_log_variance_clipped_ap  = extract(self.posterior_log_variance_clipped, self.idx[(data.t-1).to(int)], xt['AP' ].shape)
        posterior_log_variance_clipped_cpu = extract(self.posterior_log_variance_clipped, self.idx[(data.t-1).to(int)], xt['CPU'].shape)
        posterior_log_variance_clipped = {  "CPU": posterior_log_variance_clipped_cpu.to(float32),
                                            "AP" : posterior_log_variance_clipped_ap .to(float32),
                                         }
        return posterior_mean, posterior_variance, posterior_log_variance_clipped



def extract(a, t, x_shape):
    assert isinstance(a, torch.Tensor), "a must be torch.Tensor"
    b, *_ = x_shape[0], x_shape[1]
    out = a[t]
    repeat_num = int(b/t.shape[0])
    return out.repeat_interleave(repeat_num, dim=0).unsqueeze(-1)
