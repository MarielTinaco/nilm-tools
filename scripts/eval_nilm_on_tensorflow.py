
import sys, os

sys.path.append("..")

from adinilm.src.tf_nilm.mains.eval_seq2point import run_main

def main():

	ret = run_main()

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
		print(f"Results {ret}")