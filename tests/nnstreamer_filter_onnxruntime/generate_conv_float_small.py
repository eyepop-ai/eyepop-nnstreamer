#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
"""Generates tests/test_models/models/conv_float_small.onnx.

A small float CNN whose every node runs on onnxruntime's CUDA execution
provider, so a session can capture a CUDA graph (the int8 QDQ mobilenet cannot).
Input "input" float32 [1, 3, 224, 224], output "output" float32 [1, 10].
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

rng = np.random.default_rng(271)


def weights(name, shape):
    return numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32) * 0.1, name)


initializers = [
    weights("w1", (16, 3, 3, 3)),
    weights("b1", (16,)),
    weights("w2", (16, 16, 3, 3)),
    weights("b2", (16,)),
    weights("w3", (10, 16)),
    weights("b3", (10,)),
]
nodes = [
    helper.make_node("Conv", ["input", "w1", "b1"], ["c1"], strides=[2, 2], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c1"], ["r1"]),
    helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], strides=[2, 2], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c2"], ["r2"]),
    helper.make_node("GlobalAveragePool", ["r2"], ["p"]),
    helper.make_node("Flatten", ["p"], ["f"]),
    helper.make_node("Gemm", ["f", "w3", "b3"], ["output"], transB=1),
]
graph = helper.make_graph(
    nodes,
    "conv_float_small",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
    initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_models", "models", "conv_float_small.onnx")
onnx.save(model, out)
print(out)
