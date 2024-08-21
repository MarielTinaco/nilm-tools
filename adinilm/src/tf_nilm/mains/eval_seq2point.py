import sys, os
import numpy as np
import tensorflow as tf

from pathlib import Path

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader

from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..models import simple_seq2point
from .. import parse_cmd
from ..losses.quantileloss import QuantileLoss 
from ..losses.multiactivationloss import MultiActivationLoss 
from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR

def run_main():
	args = parse_cmd.get_parser().parse_args()

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = Path(args.dataset_profile) if args.dataset_profile is not None else PROFILES_DIR / "unetnilm_ukdale_20240321_155419"
	CHECKPOINT = Path(args.checkpoint)
	BATCH_SIZE = args.batch_size

	x = np.load(PROFILE_PATH / "training" / "noise_inputs.npy")
	y = np.load(PROFILE_PATH / "training" / "states.npy")
	z = np.load(PROFILE_PATH / "training" / "targets.npy")

	indices = np.arange(x.shape[0])

	data = NILMSeq2PointDataset(x, (y, z), seq_len=SEQ_LEN, indices=indices,
						             batch_size=BATCH_SIZE,
							     sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)
	
	model = tf.keras.models.load_model(CHECKPOINT)

	ret = model.evaluate(data, batch_size=BATCH_SIZE)

	return ret
