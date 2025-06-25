import json
import sys
import warnings

try:
    import torch
except Exception as e:
    sys.stderr.write("PyTorch not installed: %s\n" % e)
    sys.exit(1)

warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    sys.stderr.write("usage: pt_to_json.py <model.pt>\n")
    sys.exit(1)

model = torch.jit.load(sys.argv[1], map_location="cpu")
state = model.state_dict()

# Preserve parameter order from PyTorch so that repeated blocks
# (e.g. layers within a transformer) are emitted in the same
# order as defined in the model.
keys = list(state.keys())

layers = []
blocks = []
for key in keys:
    if key.endswith(".weight"):
        name = key[:-7]
        weight = state[key].cpu().tolist()
        bias = state.get(name + ".bias")
        bias_list = bias.cpu().tolist() if bias is not None else []
        layers.append({"name": name, "weight": weight, "bias": bias_list})

        # Detect repeating layer groups such as layers.0 or layer1 and
        # store them so the loader can recreate the original ordering.
        parts = name.split(".")
        prefix = None
        if parts[0].startswith("layers"):
            if len(parts) > 1:
                prefix = parts[0] + "." + parts[1]
        elif parts[0].startswith("layer"):
            prefix = parts[0]
        if prefix and prefix not in blocks:
            blocks.append(prefix)

print(json.dumps({"layers": layers, "blocks": blocks}))
