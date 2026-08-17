import math
from typing import Any

import torch
from   torch import nn, Tensor

from torch_geometric.nn import MessagePassing, HeteroConv, GATConv, GCNConv, GraphConv

from torch_geometric.data import HeteroData
from torch_scatter import scatter_softmax
from typing_extensions import deprecated

class EdgeAwareConv(MessagePassing):
    def __init__(self, 
                 in_channels, out_channels, 
                 edge_dim, hidden_dim,

                 act='SiLU',
                 layer_norm=True,
                 dropout: float=0.0,
                 aggr='mean'

                 ):
        """
        Message Passing
        """
        super(EdgeAwareConv, self).__init__(aggr=aggr)

        self.get_msg =  MLP([in_channels + edge_dim, hidden_dim],layer_norm=layer_norm,dropout=0.0)

        self.att_mlp  = nn.Linear(hidden_dim , 1)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.update_layer = nn.Linear(hidden_dim, out_channels)# MLP([hidden_dim, out_channels ],act=['SiLU'],layer_norm=layer_norm,dropout=dropout)# nn.Linear(out_channels + out_channels, out_channels)

        self.cutshort = nn.Linear(in_features=in_channels, out_features=out_channels) if (in_channels!=out_channels) else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Tuple[source_node_features, target_node_features]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
        Returns:
            updated vertex frature after message passing
        """


        x_src     = x[0]
        x_dst     = x[1]
        edge_attr = edge_attr

        self._x_dst_cache = x[1]

        # propagate
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)

        return out

    def message(self, x_j, x_i, edge_attr,index):
        """
        Args:
            x_j : source vertex
            x_i : target vertex
            edge_attr : edge feature
            index : Index of subgraphs in the spliced large image 
        """

        data_cat  = torch.cat([x_j, edge_attr], dim=-1) 
        msg       = self.get_msg(data_cat)
        att_score = self.LeakyReLU(self.att_mlp(msg)).squeeze(-1)

        att_weight = scatter_softmax(att_score, index) 

        return msg * att_weight.unsqueeze(-1)
    
    def update(self, aggr_out):
        x_dst = self._x_dst_cache
        out = self.update_layer(aggr_out)

        out += self.cutshort(x_dst)  # residual
        return out

class Hetero_Conv(nn.Module):
    """
    Hetero Graph Conv
    """
    def __init__(self, 
                 in_channels, out_channels, 
                 edge_dim, 
                 aggr = 'mean', 
                 act  = 'SiLU', 
                 layer_norm = True, 
                 dropout = 0.0, 
                 hidden_dim = 16, 
                 *args, **kwargs):
        super(Hetero_Conv, self).__init__(*args, **kwargs)

        self.cutshort    = nn.Linear (1 + in_channels, in_channels)
        self.cutshortue  = nn.Linear (1 + in_channels, in_channels)
        self.cutshortcpu = nn.Linear (in_channels, out_channels)
        
        self.conv_ap_ue = HeteroConv({
                                    # 
                                    ('AP',  'connect'    , 'UE'  ): EdgeAwareConv(in_channels, out_channels, edge_dim, aggr=aggr\
                                                                                  , hidden_dim=hidden_dim,act=act,layer_norm=layer_norm,dropout=dropout),
                                    ('UE',  'rev_connect', 'AP'  ): EdgeAwareConv(in_channels, out_channels, edge_dim, aggr=aggr\
                                                                                  , hidden_dim=hidden_dim,act=act,layer_norm=layer_norm,dropout=dropout),
                                      }, aggr=aggr)

        self.conv_cpu_ap = HeteroConv({
                                    ('CPU',  'connect'    , 'AP'  ): EdgeAwareConv(in_channels, in_channels, edge_dim, aggr=aggr\
                                                                                   , hidden_dim=hidden_dim,act=act,layer_norm=layer_norm,dropout=dropout),
                                }, aggr=aggr)


    def forward(self, data:HeteroData)->HeteroData:
        x = data.clone()

        # 2.1 AP Condition Embedding
        cond_AP = data['AP' ].P_max# self.cond_ap (data['AP' ].P_max)
        temp = torch.cat((x['AP'].x, cond_AP),dim=-1)
        x['AP'].x = self.cutshort(temp)

        # 2.2 UE Condition Embedding
        receive_noise = data['UE' ].receive_noise# self.rn_mlp     (data['UE' ].receive_noise)
        temp2 = torch.cat((x['UE'].x, receive_noise),dim=-1)
        x['UE'].x = self.cutshortue(temp2)

        # TODO Message passing
        conv_result = self.conv_cpu_ap(x.x_dict, x.edge_index_dict, x.edge_attr_dict)

        x['AP' ].x = conv_result['AP' ]#   + data['AP' ].x

        conv_result = self.conv_ap_ue(x.x_dict, x.edge_index_dict, x.edge_attr_dict)
        x['AP' ].x = conv_result['AP' ]#   + data['AP' ].x
        x['UE' ].x = conv_result['UE' ]
        x['CPU'].x = self.cutshortcpu(data['CPU'].x)

        return x

class ResBlock(nn.Module):
    """
    """
    def __init__(self, 
                 in_channels, hidden_channels, out_channels, 
                 act='SiLU', layer_norm=False, dropout=0,
                 has_attn = True,
                 nhead=3, d_k=64,
                 *args, **kwargs):
        super(ResBlock, self).__init__(*args, **kwargs)

        self. ue_layer1 = MLP([in_channels,in_channels], act='SiLU', layer_norm=True)
        self. ap_layer1 = MLP([in_channels,in_channels], act='SiLU', layer_norm=True)

        self. conv_ue1 = MLP([in_channels,hidden_channels], act='SiLU', layer_norm=True)
        self. conv_ap1 = MLP([in_channels,hidden_channels], act='SiLU', layer_norm=True)

        self. time_emb = TimeEmbedding(hidden_channels)
        self. time_mlp = MLP([hidden_channels, hidden_channels],act=act,layer_norm=False, dropout=0)
        self. cond_ap  = MLP([1, hidden_channels], act=act, layer_norm=False, dropout=0)

        self. ue_layer2  = MLP([hidden_channels, hidden_channels ], act='SiLU', layer_norm=True, dropout=0.0)
        self. ap_layer2  = MLP([hidden_channels, hidden_channels ], act='SiLU', layer_norm=True, dropout=0.0)
        self. ap_layer22 = MLP([hidden_channels, hidden_channels ], act='SiLU', layer_norm=True, dropout=0.0)

        self. conv_ue2   = MLP([hidden_channels,out_channels], act='SiLU', layer_norm=True)
        self. conv_ap2   = MLP([hidden_channels,out_channels], act='SiLU', layer_norm=True)

        self.cutshort_ap  = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.cutshort_cpu = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.cutshort_ue  = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        self.selfattn_ap  = SelfAttention(out_channels,nhead) if has_attn else Identity()
        self.selfattn_ue  = SelfAttention(out_channels,nhead) if has_attn else Identity()

    def forward(self, data:HeteroData):
        result = data.clone()
        # Time Embedding
        t = self.time_mlp(self.time_emb(data.t))
        # Vertex Condition Information 
        cond_ap = self.cond_ap(data['AP'].P_max)
        
        ue_result, ap_result = self.conv_ue1(self.ue_layer1(data['UE'].x)), self.conv_ap1 (self. ap_layer1(data['AP' ].x))

        ue_result, ap_result =  self.ue_layer2(ue_result + t.repeat_interleave(data['UE'].ptr[1]-data['UE'].ptr[0],dim=0).expand(-1, ap_result.shape[-1])),\
                                self.ap_layer2(ap_result + t.repeat_interleave(data['AP'].ptr[1]-data['AP'].ptr[0],dim=0).expand(-1, ap_result.shape[-1])) + cond_ap



        ue_result, ap_result = self.conv_ue2(ue_result), self.conv_ap2 (ap_result )
        # Residual Connection
        ue_result, ap_result = self.cutshort_ue(data['UE'].x) + ue_result, self.cutshort_ap(data['AP'].x) + ap_result
        # Self-Attention
        ue_result, ap_result = self.selfattn_ue(ue_result, batch_size=int(data['UE'].batch.max()+1)),self.selfattn_ap(ap_result, batch_size=int(data['UE'].batch.max()+1))

        result['UE'].x , result['AP'].x = ue_result, ap_result

        # Dimensional matching
        result['CPU'].x = self.cutshort_cpu(data['CPU'].x) # 
        return result

class ResBlockStart(nn.Module):
    """
    """
    def __init__(self, 
                 in_channels, hidden_channels, out_channels, 
                 act='SiLU', layer_norm=False, dropout=0,
                 has_attn = True,
                 nhead=3, d_k=64,
                 *args, **kwargs):
        super(ResBlockStart, self).__init__(*args, **kwargs)

        self. ue_layer1 = MLP([in_channels,in_channels], act='SiLU', layer_norm=True)
        self. ap_layer1 = MLP([in_channels,in_channels], act='SiLU', layer_norm=True)

        self. conv_ue1 = MLP([in_channels,hidden_channels], act='SiLU', layer_norm=True)
        self. conv_ap1 = MLP([in_channels,hidden_channels], act='SiLU', layer_norm=True)

        self. time_emb = TimeEmbedding(hidden_channels)
        self. time_mlp = MLP([hidden_channels, hidden_channels],act=act,layer_norm=False, dropout=0)
        self. cond_ap  = MLP([1, hidden_channels], act=act, layer_norm=False, dropout=0)

        self. ue_layer2  = MLP([hidden_channels, hidden_channels ], act='SiLU', layer_norm=True, dropout=0.0)
        self. ap_layer2  = MLP([hidden_channels, hidden_channels ], act='SiLU', layer_norm=True, dropout=0.0)
        self. ap_layer22 = MLP([hidden_channels, hidden_channels ], act='SiLU', layer_norm=True, dropout=0.0)

        self. conv_ue2   = MLP([hidden_channels,out_channels], act='SiLU', layer_norm=True)
        self. conv_ap2   = MLP([hidden_channels,out_channels], act='SiLU', layer_norm=True)

        self.cutshort_ap  = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.cutshort_cpu = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.cutshort_ue  = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        self.selfattn_ap  = SelfAttention(out_channels,nhead) if has_attn else Identity()
        self.selfattn_ue  = SelfAttention(out_channels,nhead) if has_attn else Identity()

    def forward(self, data:HeteroData):
        result = data.clone()
        # Time Embedding
        t = self.time_mlp(self.time_emb(data.t))
        # Vertex Condition Information 
        cond_ap = self.cond_ap(data['AP'].P_max)
        
        ue_result, ap_result = self.conv_ue1(self.ue_layer1(data['UE'].x)), self.conv_ap1 (self. ap_layer1(data['AP' ].x))

        ue_result, ap_result =  self.ue_layer2(ue_result + t.repeat_interleave(data['UE'].ptr[1]-data['UE'].ptr[0],dim=0).expand(-1, ap_result.shape[-1])),\
                                self.ap_layer2(ap_result + t.repeat_interleave(data['AP'].ptr[1]-data['AP'].ptr[0],dim=0).expand(-1, ap_result.shape[-1])) + cond_ap



        ue_result, ap_result = self.conv_ue2(ue_result), self.conv_ap2 (ap_result )
        # Residual Connection
        ue_result, ap_result = self.cutshort_ue(data['UE'].x) + ue_result, self.cutshort_ap(data['AP'].x) + ap_result
        # Self-Attention
        ue_result, ap_result = self.selfattn_ue(ue_result, batch_size=int(data['UE'].batch.max()+1)),self.selfattn_ap(ap_result, batch_size=int(data['UE'].batch.max()+1))

        result['UE'].x , result['AP'].x = ue_result, ap_result

        # Dimensional matching
        result['CPU'].x = self.cutshort_cpu(data['CPU'].x) # 
        return result

class PreProcessing(nn.Module):
    
    def __init__(self, dim_cpu, dim_ap, dim_ue, hidden_dim, 
                 dim_edge_access, dim_edge_fronthaul, 
                 act: str = 'Mish', layer_norm: bool = True, dropout: float = 0.1, 
                *args, **kwargs): 
        """
        """
        super(PreProcessing, self).__init__(*args, **kwargs)

        self.cpu_encoder  = MLP([dim_cpu     , hidden_dim*2, hidden_dim],['SiLU','Mish'],layer_norm, dropout)# nn.Linear(dim_cpu, hidden_dim)
        self.ap_encoder   = MLP([dim_ap      , hidden_dim*2, hidden_dim],['SiLU','Mish'],layer_norm, dropout)# nn.Linear(dim_ap , hidden_dim)
        self.ue_encoder   = MLP([dim_ue      , hidden_dim*2, hidden_dim],['SiLU','Mish'],layer_norm, dropout)# nn.Linear(dim_ue , hidden_dim)

        self.access_encoder     = MLP([dim_edge_access   , hidden_dim, hidden_dim], act=['ReLU', 'SiLU'], layer_norm=True)# nn.Linear(dim_edge_access     , hidden_dim)
        self.fronthaul_encoder  = MLP([dim_edge_fronthaul, hidden_dim, hidden_dim], act=['ReLU', 'SiLU'], layer_norm=True)# nn.Linear(dim_edge_fronthaul  , hidden_dim)

    def forward(self, data:HeteroData)->HeteroData:
        result = data.clone()

        result['AP'  ] .x   = self.ap_encoder   (data['AP' ]  .x)
        result['UE'  ] .x   = self.ue_encoder   (data['UE' ]  .x)
        result['CPU' ] .x   = self.cpu_encoder  (data['CPU' ] .x)

        result['AP' ,'connect'    ,'UE' ].edge_attr = self.access_encoder   (data['AP' ,'connect'    ,'UE' ] .edge_attr)
        result['UE' ,'rev_connect','AP' ].edge_attr = self.access_encoder   (data['UE' ,'rev_connect','AP' ] .edge_attr)
        result['CPU','connect'    ,'AP' ].edge_attr = self.fronthaul_encoder(data['CPU','connect'    ,'AP' ] .edge_attr)
        result['AP' ,'rev_connect','CPU'].edge_attr = self.fronthaul_encoder(data['AP' ,'rev_connect','CPU'] .edge_attr)

        return result

class MLP(nn.Module):
    def __init__(self, channels, act:list='SiLU', layer_norm=True, dropout=0.0):
        """
        MLP Function
        Args:
            channels: Dimention Mapping Process
            act : Act Func in Each Layer
            layer_norm : perform layer normalization?
            dropout : dropout coef
        """
        super().__init__()
        layers = []

        if isinstance(act, str):
            act_list = [act] * (len(channels) - 1)
        elif isinstance(act, list):
            if len(act) != len(channels) - 1:
                raise ValueError("""When using multiple activation functions, the length of the act list should be len (channels) -1""")
            act_list = act
        else:
            raise TypeError("The act parameter should be str or List [str]")
        
        if len(channels)==1:
            if layer_norm: 
                layers.append(nn.LayerNorm(channels[0]))
            if act == 'Mish':
                layers.append(nn.Mish())
            elif act == 'ReLU':
                layers.append(nn.ReLU())
            elif act == 'SiLU':
                layers.append(nn.SiLU())
            elif act == 'Sigmoid':
                layers.append(nn.Sigmoid())
            elif act == 'LeakyReLU':
                layers.append(nn.LeakyReLU(0.2))
            else:
                raise ValueError(f"""Unsupported activation function: {act}, should be one of ['Mash ',' ReLU ',' SiLU ',' Sigmoid ']""")
            
        else:
            if layer_norm:
                # layers.append(nn.BatchNorm1d(channels[0]))
                layers.append(nn.LayerNorm(channels[0]))
            
            for i in range(1, len(channels)):
                if act_list[i - 1] == 'Mish':
                    layers.append(nn.Mish())
                elif act_list[i - 1] == 'ReLU':
                    layers.append(nn.ReLU())
                elif act_list[i - 1] == 'SiLU':
                    layers.append(nn.SiLU())
                elif act_list[i - 1] == 'Sigmoid':
                    layers.append(nn.Sigmoid())
                elif act[i - 1] == 'LeakyReLU':
                    layers.append(nn.LeakyReLU(0.2))
                else:
                    raise ValueError(f"""Unsupported activation function: {act}, should be one of ['Mash ',' ReLU ',' SiLU ',' Sigmoid ']""")
                layers.append(nn.Linear(channels[i - 1], channels[i]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        """
            Args:
                input : Timestep t
            Returns:
                emb : output
        """
        device = input.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = input[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb 

class gPool(nn.Module):
    """
    Graph Pool Function
    """
    def __init__(self, in_dim, ratio_dict=None):
        """
        :param in_dim: input dimention
        :param ratio_dict: dict, the pool ratio
        """
        super(gPool, self).__init__()

        if ratio_dict is None:
            ratio_dict = {'AP': 0.5, 'UE': 1.0}
        self.ratio_dict = ratio_dict

        self.score_fn = nn.ModuleDict({
            node_type: nn.Linear(in_dim, 1)
            for node_type in self.ratio_dict
        })

    def forward(self, data: HeteroData):
        new_data = HeteroData()
        node_masks = {}
        perm_dict = {}

        for node_type, x in data.x_dict.items():
            ratio = self.ratio_dict.get(node_type, 1.0)
            batch = data[node_type].batch

            if ratio >= 1.0:
                # Keep All
                num_nodes = x.size(0)
                perm = torch.arange(num_nodes, device=x.device)
                mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
            else:
                # Compute the Score and Select with Top-K
                score = self.score_fn[node_type](x).squeeze(-1)
                num_graphs = batch.max().item() + 1

                perm_list = []
                mask = torch.zeros_like(score, dtype=torch.bool)

                for i in range(num_graphs):
                    idx = (batch == i).nonzero(as_tuple=False).view(-1)
                    score_i = score[idx]
                    k = max(1, int(ratio * idx.size(0)))
                    topk_idx = idx[torch.topk(score_i, k).indices]
                    perm_list.append(topk_idx)
                    mask[topk_idx] = True

                perm = torch.cat(perm_list, dim=0)

            perm_dict[node_type] = perm
            node_masks[node_type] = mask

            new_data[node_type].x = data[node_type].x[perm]
            new_data[node_type].batch = data[node_type].batch[perm]
            new_data[node_type].ptr = self.get_ptr_from_batch(new_data[node_type].batch)
            for key in data[node_type].keys():
                if key not in ['x', 'batch', 'ptr']:
                    new_data[node_type][key] = data[node_type][key][perm]

        if hasattr(data, 't'):
            new_data.t = data.t

        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_mask = node_masks[src_type]
            dst_mask = node_masks[dst_type]

            src_idx = torch.nonzero(src_mask).view(-1)
            dst_idx = torch.nonzero(dst_mask).view(-1)

            src_map = -torch.ones(src_mask.size(0), dtype=torch.long, device=src_mask.device)
            dst_map = -torch.ones(dst_mask.size(0), dtype=torch.long, device=dst_mask.device)
            src_map[src_idx] = torch.arange(src_idx.size(0), device=src_mask.device)
            dst_map[dst_idx] = torch.arange(dst_idx.size(0), device=dst_mask.device)

            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            edge_mask = src_mask[src_nodes] & dst_mask[dst_nodes]

            sub_edge_index = edge_index[:, edge_mask]
            sub_edge_index[0] = src_map[sub_edge_index[0]]
            sub_edge_index[1] = dst_map[sub_edge_index[1]]

            new_data[edge_type].edge_index = sub_edge_index

            if 'edge_attr' in data[edge_type]:
                new_data[edge_type].edge_attr = data[edge_type].edge_attr[edge_mask]

        return new_data, perm_dict

    def get_ptr_from_batch(self, batch: torch.Tensor) -> torch.Tensor:
        count = torch.bincount(batch)
        ptr = torch.cat([torch.tensor([0], device=batch.device), count.cumsum(dim=0)])
        return ptr

class gUnPool(nn.Module):
    """
    Graph UnPool Function
    """
    def __init__(self, mode='zeros', 
                 *args, **kwargs):
        """
        mode: recover method: ['zeros', 'interp', 'copy']
        - 'zeros': Zero filling for vertices that have not been reserved
        - 'interp': Future interpolation implementation
        - 'copy': Keep the vertices and copy it to its original location
        """
        super(gUnPool, self).__init__()
        self.mode = mode

    def forward(self, data_orig: HeteroData, data_pooled: HeteroData, perm_dict: dict):
        """
        data_orig: Original Graph
        data_pooled: Pooled Graph
        perm_dict: The index retained for each type of node, key=vertex Type, value=perm Tensor
        """

        data_recovered = data_orig.clone()

        for ntype in data_orig.node_types:
            if ntype not in perm_dict:
                raise ValueError(f"Missing perm for node type '{ntype}'")
            perm = perm_dict[ntype]
            x_orig = data_orig[ntype].x
            x_pooled = data_pooled[ntype].x

            # Restore node features: Insert pooled features back into the position specified by perm
            recovered_x = torch.zeros_like(x_orig)
            recovered_x[perm] = x_pooled

            data_recovered[ntype].x = recovered_x

        # Edge, etc. default to retaining data_orig
        return data_recovered

class SelfAttention(nn.Module):

    def __init__(self, in_dim, n_heads=1, d_k=None):
        """
        :param in_dim: The input vector dimension.
        :param n_heads: The number of heads in multi-head attention.
        :param d_k: The number of dimensions in each head.
        """
        super(SelfAttention, self).__init__()

        # Default d_k
        if d_k is None:
            d_k = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.projection = nn.Linear(in_dim, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, in_dim)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k
        self.cutshort = nn.Linear(in_dim, d_k) if in_dim!=d_k else nn.Identity()

    def forward(self, x, batch_size, t=None, *args, **kwargs):
        """
        :param x: (batch_size, in_dim)
        :param t: (batch_size, time_dim)
        :return: (batch_size, in_dim)
        """
        x = self.norm(x)
        _ = t
        # Get shape
        batch_size, node_num, in_dim = batch_size, x.shape[0]//batch_size, x.shape[-1]                                            # 
        # Change x to shape (batch_size, n_channels=1, in_dim)              
        x = x[:, None, :]                                                                                            # 
        # Get query, key, and values (concatenated) and shape it to (batch_size, seq, n_heads, 3 * d_k)             
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)                                       # (batch_size, seq, n_heads, 3 * d_k)
        # Split query, key, and values. Each of them will have shape (batch_size, seq, n_heads, d_k)                
        q, k, v = torch.chunk(qkv, 3, dim=-1)                                                                           # 
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale                                                       # 
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$

        attn = attn.softmax(dim=-1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to (batch_size, seq, n_heads * d_k)
        res = res.reshape(-1, 1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Squeeze to shape (batch_size, in_dim)
        res = torch.squeeze(res, dim=-2)
        return res

class Identity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor, batch_size) -> Tensor:
        return input
