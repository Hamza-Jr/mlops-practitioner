from pathlib import Path

import onnx

from prodml.utils.config import settings


def _format_shape(tensor_type: onnx.TypeProto.Tensor) -> tuple:
    """Convert ONNX tensor dimensions to a readable shape."""
    shape = []

    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append("?")

    return tuple(shape)


def _format_type(tensor_type: onnx.TypeProto.Tensor) -> str:
    """Return the ONNX tensor data type name."""
    if tensor_type.elem_type == 0:
        return "UNDEFINED"

    return onnx.TensorProto.DataType.Name(tensor_type.elem_type)


def validate_onnx_model(model_path: Path | None = None) -> None:
    """Validate and inspect the ONNX model artifact."""

    model_path = model_path or (settings.artifacts_dir / f"{settings.model_name}.onnx")

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    model = onnx.load(model_path)

    # Structural validation.
    onnx.checker.check_model(model)

    print(f"ONNX model: {model_path}")
    print(f"Producer: {model.producer_name}")
    print(f"IR version: {model.ir_version}")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    print("\nInputs:")
    print(f"  Count: {len(model.graph.input)}")

    for input_tensor in model.graph.input:
        tensor_type = input_tensor.type.tensor_type

        print(f"  Name: {input_tensor.name}")
        print(f"  Type: {_format_type(tensor_type)}")
        print(f"  Shape: {_format_shape(tensor_type)}")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    print("\nOutputs:")
    print(f"  Count: {len(model.graph.output)}")

    for output_tensor in model.graph.output:
        tensor_type = output_tensor.type.tensor_type

        print(f"  Name: {output_tensor.name}")
        print(f"  Type: {_format_type(tensor_type)}")
        print(f"  Shape: {_format_shape(tensor_type)}")

    print("\nONNX model is valid.")


if __name__ == "__main__":
    validate_onnx_model()
