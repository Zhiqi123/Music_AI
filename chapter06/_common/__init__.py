"""第六章通用工具：硬件检测、绘图、数据加载等。

6.1 阶段暴露 device_utils 与 plot_utils；其余模块（train、config、
early_stopping、evaluation、tensorboard_utils）将在 6.2 写作时随训练模板一同定型。
"""
from .device_utils import get_device, get_num_workers
from .env_check import CheckResult, PRESETS, check_environment
from .plot_utils import (
    CNAME_DISPLAY_REMAP,
    add_recall_colorbar,
    display_cname,
    plot_confusion_matrix,
    setup_chinese_font,
)

__all__ = [
    "get_device",
    "get_num_workers",
    "setup_chinese_font",
    "plot_confusion_matrix",
    "add_recall_colorbar",
    "display_cname",
    "CNAME_DISPLAY_REMAP",
    "check_environment",
    "CheckResult",
    "PRESETS",
]
