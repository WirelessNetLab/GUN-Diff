from .communication import location_generate  as lg
from .communication import channel_generate   as cg
from .communication import channel_estimate   as ce
from .communication import pilot_generate     as pg
from .communication import pilot_allocate     as pa
from .communication import get_wired_fronthaul_noise
from .communication import get_wireless_fronthaul_noise

from .communication import device_show    as ds
from .communication import robust_show as rls
from .communication import get_precode_matrix_batch as gpmb
from .communication import data_in_BS as diB
from .communication import data_transmit as dt
from .communication import get_global_SE as ggSE
from .communication import get_SE as getSE

from .Algorithm     import get_expert_dcp #, get_expert_sca, get_expert_wmmse

