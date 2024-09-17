

import sys, os

sys.path.append("..")

from adinilm.src.tf_nilm.mains.tflite_micro_train import run_main
from adinilm.src.tf_nilm import parse_cmd


def main():

	args = parse_cmd.get_parser().parse_args()
	ret = run_main(args)

	return ret

if __name__ == "__main__":
	ret = None
	try:
		ret = main()
	except KeyboardInterrupt:
		print("\n-- KeyboardInterrupt --")
	# except Exception as e:
	# 	print(str(e))
	finally:
		if ret is not None:
			print(f"Log for this session: {ret['logs']}")