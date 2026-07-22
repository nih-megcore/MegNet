#!/usr/bin/env bash
#
# convert_keras_model.sh
#
# Converts a legacy Keras/TensorFlow SavedModel directory
# (containing assets/, keras_metadata.pb, saved_model.pb, variables/)
# into a Keras 3 native ".keras" model file.
#
# Usage:
#   ./convert_keras_model.sh /path/to/model_folder [output_name.keras]
#
# Requirements:
#   - A Python environment where the model can be loaded, typically
#     tf-keras (legacy Keras 2 shim) since raw SavedModel loading isn't
#     supported directly by keras>=3. tf-keras is used specifically to
#     load the old format before re-saving in the new one.
#
#     pip install tensorflow tf-keras keras
#
set -euo pipefail

# ---- Argument parsing -------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_saved_model_dir> [output_name.keras]" >&2
    exit 1
fi

MODEL_DIR="$1"
OUTPUT_NAME="${2:-converted_model.keras}"
# Derive the .h5 sibling name from OUTPUT_NAME (same basename, .h5 extension)
OUTPUT_BASENAME="${OUTPUT_NAME%.keras}"
OUTPUT_H5_NAME="${OUTPUT_BASENAME}.h5"

# ---- Sanity checks ------------------------------------------------------
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: '$MODEL_DIR' is not a directory." >&2
    exit 1
fi

if [ ! -f "$MODEL_DIR/saved_model.pb" ]; then
    echo "Error: '$MODEL_DIR' does not look like a TensorFlow SavedModel" \
         "(missing saved_model.pb)." >&2
    exit 1
fi

if [ ! -d "$MODEL_DIR/variables" ]; then
    echo "Error: '$MODEL_DIR' does not look like a TensorFlow SavedModel" \
         "(missing variables/ directory)." >&2
    exit 1
fi

# Resolve to an absolute path so the Python snippet is unambiguous.
MODEL_DIR_ABS="$(cd "$MODEL_DIR" && pwd)"
OUTPUT_PATH="$(pwd)/$OUTPUT_NAME"
OUTPUT_H5_PATH="$(pwd)/$OUTPUT_H5_NAME"

echo "Source SavedModel dir : $MODEL_DIR_ABS"
echo "Output Keras 3 file   : $OUTPUT_PATH"
echo "Output H5 file        : $OUTPUT_H5_PATH"
echo

# ---- Check for required Python packages --------------------------------
python3 - <<'PYCHECK'
import importlib
import importlib.util
import sys

missing = []
for pkg in ("tensorflow", "tf_keras", "keras"):
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)

if missing:
    print(f"Missing required Python packages: {', '.join(missing)}", file=sys.stderr)
    print("Either activate an environment that already has these installed,", file=sys.stderr)
    print("or install them with:", file=sys.stderr)
    print("    pip install tf-keras keras", file=sys.stderr)
    sys.exit(1)
PYCHECK

# ---- Run the actual conversion ------------------------------------------
python3 - "$MODEL_DIR_ABS" "$OUTPUT_PATH" "$OUTPUT_H5_PATH" <<'PYCONVERT'
import sys
import os

model_dir = sys.argv[1]
output_path = sys.argv[2]
output_h5_path = sys.argv[3]

# tf_keras provides the legacy Keras 2 loader capable of reading
# TF SavedModel-format models (with keras_metadata.pb).
import tf_keras as legacy_keras

# Skip compiling the model on load. Compilation requires reconstructing
# the optimizer/loss/metrics (e.g. custom TF Addons objects), which we
# don't need just to migrate the architecture + weights to Keras 3.
print(f"Loading legacy SavedModel from: {model_dir}")
legacy_model = legacy_keras.models.load_model(model_dir, compile=False)
print("Model loaded successfully (compile=False).")

# The object above is a tf_keras.Functional/Sequential instance. Saving it
# directly with .save() embeds 'module': 'tf_keras.src.engine...' in the
# config, which Keras 3 cannot deserialize (it only knows 'keras.*' /
# 'keras.src.*' module paths). To produce a genuinely native Keras 3
# model, we rebuild the architecture using the real `keras` package from
# the legacy model's config, patching module paths, then transfer weights.
import keras
import json

print(f"\nRebuilding architecture as native Keras 3 model "
      f"(keras version: {keras.__version__})...")

legacy_config = legacy_model.get_config()

def _fix_modules(obj):
    """Recursively rewrite tf_keras module/class refs to native keras ones
    so keras 3's deserializer can resolve every layer/object in the config.
    Also fixes known config-shape differences between tf_keras and keras 3
    (e.g. BatchNormalization's `axis` stored as a list instead of an int)."""
    if isinstance(obj, dict):
        if obj.get("module", "").startswith("tf_keras"):
            obj["module"] = "keras.layers" if "layers" in obj.get("module", "") else "keras"
        if obj.get("class_name") == "Functional":
            obj["module"] = "keras"
            obj["registered_name"] = None
        if obj.get("class_name") == "BatchNormalization":
            axis = obj.get("config", {}).get("axis")
            if isinstance(axis, list) and len(axis) == 1:
                obj["config"]["axis"] = axis[0]
        for v in obj.values():
            _fix_modules(v)
    elif isinstance(obj, list):
        for item in obj:
            _fix_modules(item)
    return obj

legacy_config = _fix_modules(legacy_config)

# Reconstruct using native Keras 3's Functional/Sequential deserializer.
if legacy_model.__class__.__name__ == "Sequential":
    model = keras.Sequential.from_config(legacy_config)
else:
    model = keras.Model.from_config(legacy_config)

# Transfer weights by name to be robust to any minor ordering differences.
model.set_weights(legacy_model.get_weights())
print("Weights transferred to native Keras 3 model.")
model.summary()

if not output_path.endswith(".keras"):
    output_path += ".keras"

model.save(output_path)
print(f"\nConversion complete. Saved Keras 3 model to: {output_path}")
print("Note: model was loaded with compile=False, so it has no optimizer/")
print("loss/metrics attached. Re-compile with model.compile(...) before training.")

# Also save a legacy HDF5 (.h5) copy. Keras 3 still supports writing the
# H5 format via the same save() call when given an .h5 extension.
if not output_h5_path.endswith(".h5"):
    output_h5_path += ".h5"

model.save(output_h5_path)
print(f"Also saved legacy H5 model to: {output_h5_path}")
PYCONVERT

echo
echo "Done. You can now load the model in Keras 3 with:"
echo "    import keras"
echo "    model = keras.models.load_model('$OUTPUT_NAME')"
echo "or the legacy H5 file with:"
echo "    model = keras.models.load_model('$OUTPUT_H5_NAME')"
