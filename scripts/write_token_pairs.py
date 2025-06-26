#!/usr/bin/env python3
import json
import sys

if len(sys.argv) < 4:
    sys.stderr.write("usage: write_token_pairs.py <tokens.txt> <seq_len> <output.jsonl>\n")
    sys.exit(1)

with open(sys.argv[1]) as f:
    ids = [int(x) for x in f.read().split()]

seq_len = int(sys.argv[2])
output = sys.argv[3]

with open(output, "w") as out:
    for i in range(len(ids) - seq_len):
        seq = [[ids[j]] for j in range(i, i + seq_len)]
        pair = [seq, [ids[i + seq_len]]]
        out.write(json.dumps(pair))
        out.write("\n")
