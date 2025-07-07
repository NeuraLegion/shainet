import sys
import warnings
try:
    import torch
except Exception as e:
    sys.stderr.write("PyTorch not installed: %s\n" % e)
    sys.exit(1)

warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    sys.stderr.write("usage: pt_forward.py <model.pt>\n")
    sys.exit(1)

model = torch.jit.load(sys.argv[1], map_location='cpu')
model.eval()
with torch.no_grad():
    inp = torch.tensor([1.0, 2.0])
    out = model(inp)
    if hasattr(out, 'tolist'):
        val = out.tolist()
        if isinstance(val, list):
            val = val[0]
    else:
        val = float(out)
    print(val)
