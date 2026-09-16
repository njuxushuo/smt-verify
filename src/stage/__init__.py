"""Stage input models, loading, validation, and dataflow analysis."""

from importlib import import_module

_EXPORTS = {
    "DeviceMeshError": (".mesh", "DeviceMeshError"),
    "DeviceMeshSpec": (".model", "DeviceMeshSpec"),
    "OpSpec": (".model", "OpSpec"),
    "ProgramAnalysisError": (".analysis", "ProgramAnalysisError"),
    "ProgramSpec": (".model", "ProgramSpec"),
    "PlacementSpec": (".model", "PlacementSpec"),
    "RelationSpec": (".model", "RelationSpec"),
    "StageInputError": (".loader", "StageInputError"),
    "StageSpec": (".model", "StageSpec"),
    "TensorSpec": (".model", "TensorSpec"),
    "coordinate_to_rank": (".mesh", "coordinate_to_rank"),
    "enumerate_coordinate_group": (".mesh", "enumerate_coordinate_group"),
    "find_program_inputs": (".analysis", "find_program_inputs"),
    "load_stage": (".loader", "load_stage"),
    "mesh_size": (".mesh", "mesh_size"),
    "rank_to_coordinate": (".mesh", "rank_to_coordinate"),
    "validate_program_dataflow": (".analysis", "validate_program_dataflow"),
    "validate_stage": (".validator", "validate_stage"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
