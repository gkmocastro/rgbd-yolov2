import argparse

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument("--echo", help="echo the string you use here", action="store_true")
args = parser.parse_args()
print(args.echo)
