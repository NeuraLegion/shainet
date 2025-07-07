import sys
import warnings
try:
    import torch
except Exception as e:
    sys.stderr.write("PyTorch not installed: %s\n" % e)
    sys.exit(1)

warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    sys.stderr.write("usage: build_simple_model.py <output.pt>\n")
    sys.exit(1)

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(2, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1),
)
example = torch.randn(1, 2)
traced = torch.jit.trace(model, example)
traced.save(sys.argv[1])
