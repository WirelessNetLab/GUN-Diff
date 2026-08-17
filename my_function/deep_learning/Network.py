import torch
from torch import nn
from .model import (PreProcessing, 
                    Hetero_Conv, 
                    ResBlock, 
                    ResBlockStart, 
                    gPool, 
                    gUnPool,
                    MLP)
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler

class GraphUNet(nn.Module):
    def __init__(self, 
                 dim_cpu, dim_ap, dim_ue, hidden_dim,
                 dropout=0.0,
                 *args, **kwargs):
        super(GraphUNet, self).__init__(*args, **kwargs)
        self.pretreat = PreProcessing(dim_cpu, dim_ap, dim_ue, hidden_dim, 1, 1)
        edge_dim = hidden_dim
        ############################################################################################################
        n_blocks = 1 
        cof1 = 0
        cof2 = 1
        res_cat = False
        gcn_cat = True
        self.res_cat = res_cat
        self.gcn_cat = gcn_cat
        dims = [hidden_dim*1, hidden_dim*1,
                hidden_dim, hidden_dim,]
        n_resolutions = len(dims)
        Kappa = 1/2
        ratio_dict= {
                        'AP': Kappa  , 
                        'UE': Kappa  , 
                        }
        ############################################################################################################
        down = []
        in_dim = hidden_dim
        self.begin_end_dim = {
                                'cpu' : dim_cpu , 
                                'ap'  : dim_ap  , 
                                'ue'  : dim_ue  , 
                              }

        self.outMLP = nn.ModuleDict({
            "UE" : MLP([hidden_dim*1],act='SiLU',layer_norm=True),
            "AP" : MLP([hidden_dim*1],act='SiLU',layer_norm=True),
        })

        self.outLayer = nn.ModuleDict({
            "CPU": MLP([hidden_dim*1, self.begin_end_dim['cpu']], act='SiLU',layer_norm=True),
            "AP" : MLP([hidden_dim*1, self.begin_end_dim['ap' ]], act='SiLU',layer_norm=True),
            "UE" : MLP([hidden_dim*1, self.begin_end_dim['ue' ]], act='SiLU',layer_norm=True),
        })


        for i in range(n_resolutions):
            out_dim = dims[i]
            down.append(Hetero_Conv(in_channels=in_dim, out_channels=in_dim, edge_dim=edge_dim, hidden_dim=in_dim, aggr='sum', ))
            down    .append(ResBlock(in_channels=in_dim, hidden_channels = in_dim, out_channels=out_dim, has_attn=True if i >=cof1 else False, nhead=cof2, dropout=dropout))
            for _ in range(n_blocks-1):
                down.append(ResBlock(in_channels=out_dim, hidden_channels = out_dim, out_channels=out_dim , has_attn=False if i >=cof1 else False, nhead=cof2, dropout=dropout))

            down.append(gPool(out_dim, ratio_dict ))
            in_dim = out_dim
            

        self.down = nn.ModuleList(down)

        self.middle = nn.ModuleList([ResBlock(in_channels=in_dim, hidden_channels = in_dim, out_channels=in_dim, has_attn=True ,  nhead=cof2, dropout=dropout),
                                     ResBlock(in_channels=in_dim, hidden_channels = in_dim, out_channels=in_dim, has_attn=False,  nhead=cof2, dropout=dropout), ])

        up = []
        for i in reversed(range( n_resolutions)):
            out_dim = dims[i-1] if i-1>=0 else hidden_dim
            up.append(gUnPool())
            up.append(Hetero_Conv(in_channels=in_dim+(in_dim if gcn_cat else 0), out_channels=in_dim, edge_dim=edge_dim, hidden_dim=in_dim, aggr='sum', ))

            for _ in range(n_blocks-1):
                up.append( ResBlock(in_channels=in_dim+(in_dim  if res_cat else 0) , hidden_channels = in_dim, out_channels=in_dim  , has_attn=False if i >=cof1 else False, nhead=cof2, dropout=dropout))
            up.append(ResBlockStart(in_channels=in_dim+(in_dim if res_cat else 0) , hidden_channels = in_dim , out_channels=out_dim , has_attn=False if i >=cof1 else False, nhead=cof2, dropout=dropout))

            in_dim = out_dim
            

        self.up = nn.ModuleList(up)

        self.norm = nn.LayerNorm(in_dim)
        self.act = nn.SiLU()
        self.scaler = StandardScaler()



    def forward(self, data:HeteroData):

        x = self.pretreat(data.clone()) # -> nhid
        unpool = []
        perm_save = []
        xs = []

        for m in self.down:
            if   isinstance(m, Hetero_Conv):
                x = m(x)
                xs.append(x)
            elif isinstance(m, ResBlock):
                x = m(x)
                xs.append(x)

            elif isinstance(m, gPool):
                x, perm = m(x)
                xs.append(x)
                perm_save.append(perm)

        # ## ### # ## ### # ## ### # ## ### # ## ### # ## ### # ## ### 
        for m in self.middle:
            x = m(x)
        # ## ### # ## ### # ## ### # ## ### # ## ### # ## ### # ## ### 

        for m in self.up:
            if isinstance(m, Hetero_Conv):

                assert x['AP'].x.shape[-2] == xs[-1]['AP'].x.shape[-2], \
                    f"Dimension mismatch: x={x['AP'].x.shape}, xs_pop={xs[-1]['AP'].x.shape}"
                if self.gcn_cat:
                    x = m(datacat(x, xs.pop()))
                else:
                    x = m(x)
                    xs.pop()
                
            elif isinstance(m, ResBlockStart):
                assert x['AP'].x.shape[-2] == xs[-1]['AP'].x.shape[-2] , "Dimension mismatch"
                if self.res_cat:
                    x = m(datacat(x, xs.pop()))
                else:
                    x = m(x)
                    xs.pop()
                
            elif isinstance(m, ResBlock):
                assert x['AP'].x.shape[-2] == xs[-1]['AP'].x.shape[-2] , "Dimension mismatch"
                if self.res_cat:
                    x = m(datacat(x, xs.pop()))
                else:
                    x = m(x)
                    xs.pop()
                
            elif isinstance(m, gUnPool):
                xs.pop() 
                assert x['AP'].x.shape[-1] == xs[-1]['AP'].x.shape[-1] , "Dimension mismatch"
                x = m(xs[-1], x, perm_save.pop())

        # Assert Symmetry
        assert  not len(xs)

        # End #############################################################################
        for key in ['AP','UE']:
            x[key].x = self.outLayer[key](self.outMLP[key](x[key].x))

        # For Reverse Diffusion  ##########################################
        x['CPU' ].x = data['CPU'].x
        for edge in x.edge_types:
            x[edge].edge_attr = data[edge].edge_attr

        x['AP'].noise_pred, x['UE'].noise_pred = x    ['AP' ].x   , x   ['UE'].x
        x['AP'].xt        , x['UE'].xt         = data ['AP' ].x   , data['UE'].x


        return x

def datacat(main_data: HeteroData, aux_data: HeteroData) -> HeteroData:
    """
    feature cat for skip-conncetion
    """
    new_data = main_data.clone()

    for node_type in main_data.node_types:
        if "x" in main_data[node_type] and "x" in aux_data[node_type]:
            # concate the final dim
            new_data[node_type].x = torch.cat(
                [main_data[node_type].x, aux_data[node_type].x], dim=-1
            )

    return new_data