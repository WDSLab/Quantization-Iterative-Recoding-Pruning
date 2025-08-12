# methods/__init__.py
from .baseline import run as baseline
from .prune_basic import run as prune_basic
from .fusion_ot import run as fusion_ot
from .sms import run as sms
from .spsrc import run as spsrc
from .structalign import run as structalign
from .geta import run as geta
from .awp import run as awp
from .edg import run as edg

from .qat import run as qat
from .qdrop import run as qdrop
from .qpsnn import run as qpsnn
from .qirp import run as qirp

METHODS = {
    "baseline": baseline,
    "prune_basic": prune_basic,
    "fusion_ot": fusion_ot,
    "sms": sms,
    "spsrc": spsrc,
    "structalign": structalign,
    "geta": geta,
    "awp": awp,
    "edg": edg,

    "qat": qat,
    "qdrop": qdrop,
    "qpsnn": qpsnn,
    "qirp": qirp,
}