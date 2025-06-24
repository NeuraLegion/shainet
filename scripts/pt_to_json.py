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
keys = list(state.keys())
keys.sort()

layers = []
for key in keys:
    if key.endswith(".weight"):
        name = key[:-7]
        weight = state[key].cpu().tolist()
        bias = state.get(name + ".bias")
        bias_list = bias.cpu().tolist() if bias is not None else []
        layers.append({"name": name, "weight": weight, "bias": bias_list})

print(json.dumps({"layers": layers}))
