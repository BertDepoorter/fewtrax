"""Time-frequency basis abstraction layer.

Provides a common interface (TFGrid) and shared deposition logic
(direct_tf_mode) that work for any concrete TF basis.  Currently
implemented bases:

- :class:`~fewtrax.utils.tf_bases.wdm.WDMGrid`  — Wilson–Daubechies–Meyer
- :class:`~fewtrax.utils.tf_bases.sft.SFTGrid`  — Short-Time Fourier Transform
"""

from fewtrax.utils.tf_bases.base import TFGrid, direct_tf_mode
from fewtrax.utils.tf_bases.wdm import (
    WDMGrid,
    default_grid,
    meyer_window,
    meyer_kernel,
)
from fewtrax.utils.tf_bases.sft import (
    SFTGrid,
    default_sft_grid,
    sft_kernel,
    sft_kernel_exact,
)

__all__ = [
    "TFGrid",
    "direct_tf_mode",
    "WDMGrid",
    "default_grid",
    "meyer_window",
    "meyer_kernel",
    "SFTGrid",
    "default_sft_grid",
    "sft_kernel",
    "sft_kernel_exact",
]
