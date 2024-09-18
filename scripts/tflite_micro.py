

import sys, os

sys.path.append("..")

from adinilm.src.tf_nilm import parse_cmd

def main():

	ret = None
	parser = parse_cmd.get_parser()
	mode_args = parser.add_argument_group("Run mode Arguments")
	mode_args.add_argument("--evaluate", action="store_true", default=False, help="Run in evaluation mode")
	
	args = parser.parse_args()

	if bool(args.evaluate):
		from adinilm.src.tf_nilm.mains.tflite_micro_eval import run_main
		run_main(args)

	else:
		from adinilm.src.tf_nilm.mains.tflite_micro_train import run_main
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