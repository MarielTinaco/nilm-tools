
import sys, os

sys.path.append("..")

from adinilm.src.tf_nilm.mains.train_seq2point import run_main

def main():

	run_main()


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n-- KeyboardInterrupt --")
	# except Exception as e:
	# 	print(str(e))
	finally:
		print("print log file here")