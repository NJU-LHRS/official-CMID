import torch
import argparse
from collections import OrderedDict


def parse_args():
	parser = argparse.ArgumentParser(description='Convert Backbone')
	parser.add_argument("--checkpoint_path", type=str)
	parser.add_argument("--out_path", type=str)
	args = parser.parse_args()

	return args


def main():
	args = parse_args()
	checkpoint_path = args.checkpoint_path
	out_path = args.out_path
	checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
	new_ckpt = OrderedDict()
	for key in checkpoint.keys():
		if key.startswith("online_encoder.model."):
			new_ckpt[key[len("online_encoder.model."):]] = checkpoint[key]
	torch.save(new_ckpt, out_path)


if __name__ == "__main__":
	main()