"""Shared admin control-plane helpers for ModelRouter."""

from hermes.plugins.model_router.admin.actions import (
    AdminActionError,
    action_descriptors,
    run_admin_action,
)
from hermes.plugins.model_router.admin.config_edit import save_proxy_config_patch
from hermes.plugins.model_router.admin.state import build_admin_state, settings_paths
from hermes.plugins.model_router.admin.supervisor import (
    ProxyProcessStatus,
    ProxyProcessSupervisor,
)

__all__ = [
    "AdminActionError",
    "ProxyProcessStatus",
    "ProxyProcessSupervisor",
    "action_descriptors",
    "build_admin_state",
    "run_admin_action",
    "save_proxy_config_patch",
    "settings_paths",
]
