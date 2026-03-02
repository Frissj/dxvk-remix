#pragma once

// RTX MegaGeo per-file verbose logging toggles.
// Set individual flags to 1 to enable logging for that file, 0 to disable.
// All flags default to 0 (disabled) for performance.

// cluster_builder
#define RTXMG_LOG_CLUSTER_ACCEL_BUILDER    1  // cluster_accel_builder.cpp

// nvrhi_adapter
#define RTXMG_LOG_DONUT_ADAPTER            0  // donut_adapter.cpp
#define RTXMG_LOG_NVRHI_DXVK_COMMAND_LIST  1  // nvrhi_dxvk_command_list.cpp
#define RTXMG_LOG_NVRHI_DXVK_DEVICE        0  // nvrhi_dxvk_device.cpp
#define RTXMG_LOG_NVRHI_SCRATCH_MANAGER    0  // nvrhi_scratch_manager.cpp

// Core
#define RTXMG_LOG_RTX_MEGAGEO_BUILDER      1  // rtx_megageo_builder.cpp

// utils
#define RTXMG_LOG_BUFFER                   1  // buffer.cpp
