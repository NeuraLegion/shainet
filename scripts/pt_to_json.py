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
    sys.stderr.write("usage: pt_to_json.py <model.pt|pytorch_model.bin>\n")
    sys.exit(1)


def convert_torchscript(path):
    model = torch.jit.load(path, map_location="cpu")
    state = model.state_dict()

    keys = list(state.keys())  # preserve parameter order

    layers = []
    blocks = []
    for key in keys:
        if key.endswith(".weight"):
            name = key[:-7]
            weight = state[key].cpu().tolist()
            bias = state.get(name + ".bias")
            bias_list = bias.cpu().tolist() if bias is not None else []
            layers.append({"name": name, "weight": weight, "bias": bias_list})

            parts = name.split(".")
            prefix = None
            if parts[0].startswith("layers"):
                if len(parts) > 1:
                    prefix = parts[0] + "." + parts[1]
            elif parts[0].startswith("layer"):
                prefix = parts[0]
            if prefix and prefix not in blocks:
                blocks.append(prefix)

    return {"layers": layers, "blocks": blocks}


def convert_hf_gpt(path):
    state = torch.load(path, map_location="cpu")

    layers = []
    blocks = []

    if "transformer.wte.weight" in state:
        emb = state["transformer.wte.weight"].cpu().tolist()
        layers.append({"name": "embedding", "weight": emb, "bias": []})

    if "lm_head.weight" in state:
        out_w = state["lm_head.weight"].cpu().tolist()
        out_b = state.get("lm_head.bias")
        out_bias = out_b.cpu().tolist() if out_b is not None else []
        layers.append({"name": "out", "weight": out_w, "bias": out_bias})

    index = 0
    while True:
        prefix = f"transformer.h.{index}"
        attn_w = state.get(prefix + ".attn.c_attn.weight")
        if attn_w is None:
            break
        blocks.append(f"layer{index}")

        attn_b = state[prefix + ".attn.c_attn.bias"]
        proj_w = state[prefix + ".attn.c_proj.weight"].t()
        proj_b = state[prefix + ".attn.c_proj.bias"]

        hidden = attn_b.shape[0] // 3
        wqkv = attn_w.t()
        w_q = wqkv[:hidden].cpu().tolist()
        w_k = wqkv[hidden:2 * hidden].cpu().tolist()
        w_v = wqkv[2 * hidden :].cpu().tolist()
        b_q = attn_b[:hidden].cpu().tolist()
        b_k = attn_b[hidden:2 * hidden].cpu().tolist()
        b_v = attn_b[2 * hidden :].cpu().tolist()

        layers.append({"name": f"layer{index}.mha.w_q", "weight": w_q, "bias": b_q})
        layers.append({"name": f"layer{index}.mha.w_k", "weight": w_k, "bias": b_k})
        layers.append({"name": f"layer{index}.mha.w_v", "weight": w_v, "bias": b_v})
        layers.append({
            "name": f"layer{index}.mha.w_o",
            "weight": proj_w.cpu().tolist(),
            "bias": proj_b.cpu().tolist(),
        })

        ffn1_w = state[prefix + ".mlp.c_fc.weight"].t()
        ffn1_b = state[prefix + ".mlp.c_fc.bias"]
        ffn2_w = state[prefix + ".mlp.c_proj.weight"].t()
        ffn2_b = state[prefix + ".mlp.c_proj.bias"]
        layers.append({
            "name": f"layer{index}.ffn.w1",
            "weight": ffn1_w.cpu().tolist(),
            "bias": ffn1_b.cpu().tolist(),
        })
        layers.append({
            "name": f"layer{index}.ffn.w2",
            "weight": ffn2_w.cpu().tolist(),
            "bias": ffn2_b.cpu().tolist(),
        })

        ln1_w = state[prefix + ".ln_1.weight"].cpu().tolist()
        ln1_b = state[prefix + ".ln_1.bias"].cpu().tolist()
        ln2_w = state[prefix + ".ln_2.weight"].cpu().tolist()
        ln2_b = state[prefix + ".ln_2.bias"].cpu().tolist()
        layers.append({"name": f"layer{index}.norm1", "weight": ln1_w, "bias": ln1_b})
        layers.append({"name": f"layer{index}.norm2", "weight": ln2_w, "bias": ln2_b})

        index += 1

    return {"layers": layers, "blocks": blocks}


path = sys.argv[1]
if path.endswith(".bin"):
    data = convert_hf_gpt(path)
else:
    data = convert_torchscript(path)

print(json.dumps(data))
