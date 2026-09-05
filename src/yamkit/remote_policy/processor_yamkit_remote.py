"""Numerical identity on the client: all saved model processing is server-side."""

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    RenameObservationsProcessorStep,
    make_policy_processor_pipelines,
)


def make_yamkit_remote_pre_post_processors(config, dataset_stats=None):
    # Never consume dataset_stats here: actions are already in robot units.
    return make_policy_processor_pipelines(
        input_steps=[RenameObservationsProcessorStep(rename_map={}),
                     AddBatchDimensionProcessorStep(), DeviceProcessorStep(device="cpu")],
        output_steps=[DeviceProcessorStep(device="cpu")],
    )
