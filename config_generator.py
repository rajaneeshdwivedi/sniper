import json
from pathlib import Path
import numpy as np
import re
from pipeline_utils import apply_override, get_leaf_elements

def generate_config(base_path, override=None):
	"""Generate configuration file with enhanced temporal parameters and feature interactions"""
	
	if override is None:
		run_id = ""
		run_dir = Path(base_path)
	else:
		leaves = get_leaf_elements(override)
		# Collate overridden parameters but exclude paths
		run_id = "_".join(f"{k}_{v}" for k, v in leaves.items() if not k.endswith('_dir'))	
		run_dir = Path(base_path) / run_id

	config = {
		"feature_generation": {
			"feature_groups": {
				"price": [
					 "open_ret",
					 "high_ret",
					#  "low_ret",
					 "close_ret"
				],
				# "volume": [
				# 	 "volume_ret"
				# ],
				"volatility": [
					# "true_range",
					"vol_atr_pct",
					# "vol_atr_acc",
					"vol_bb_width",
					"vol_regime"
				],
				# "swing": [
				# 	 "bars_since_swing_low",
				# 	 "bars_since_swing_high"
				# ],
				# "volume_relative": [
				# 	 "vol_rel_short",
				# 	 "vol_rel_long", 
				# 	 "vol_trend"
				# ],
				"vwap": [
					"vwap_support",
					"vwap_resistance",
					"vwap_low",   
					"vwap_high",  
				]
			},
			"source_width": 500,
			"feature_width": 20,
			"label_params": {
				"atr_target_mult": 2.5,
				"atr_stop_mult": 2.5,
				"atr_period": 14
			}
		},
		"vwap_params": {
			"left_pad": 15,
			"right_pad": 2
		},
		"loss_params": {
			"smooth_factor": 0.15,  # Reduced from 0.3 for less aggressive smoothing
			"l2_weight": 0.004,  # Reduced regularization
			"smooth_factor": 0.2,
			"confidence_weight": 0.18,
			"focal_gamma": 0.8,
			"asymmetry_factor": 0.9,
			"calibration_weight": 0.2,
			"precision_weight": 0.05,
			"high_conf_threshold": 0.7,			
		},
		"model_params": {
			"hidden_dim": 92,  # Increased from 32 for better expressiveness
			"dropout": 0.25,  # Reduced dropout rate
			"use_layernorm": True,  # Enable layer normalization
			"confidence_range": [0.0, 1.0],  # Wider confidence range
			"prediction_range": [0.0, 1.0],  # Wider prediction range
			"feature_crossing": True
		},
		"training_params": {
			'learning_rate': 1e-5,  # Slightly more conservative learning rate
			"min_learning_rate": 1e-6,
			"batch_size": 128,  # Larger batch size for stability
			"num_epochs": 200,
			"patience": 15,  # Increased patience
			"min_delta": 0.0001,  # Reduced minimum delta for early stopping

			'confidence_lr_multiplier': 1.0,  # Reduced multiplier for confidence branch
			'weight_decay': 0.01,  # Keep L2 regularization
			'scheduler_type': 'cosine_warmup',
			'warmup_epochs': 5,  # Increased warmup period
			'AdamW_beta1': 0.9,
			'AdamW_beta2': 0.999,
			"gradient_clip": 0.5,  # Increased gradient clipping threshold
			"use_lr_scheduler": True,
			"use_swa": False,
			"swa_start": 10,
			"use_swa_scheduler": True,
			"swa_lr": 0.001,		  
			"swa_anneal_epochs": 5  
		},
		"dataset_params": {
			"mono_dataset": {
				"codes": [
					'BTCUSDT'
				]
			},
			"small_dataset": {
				"codes": [
					"BTCUSDT",
					"DOTUSDT",
					"GALAUSDT"
				]
			},

			"medium_dataset": {
				"codes": [
					"BTCUSDT",
					"ETHUSDT",
					"SOLUSDT",
					"LINKUSDT",
					"SHIBUSDT"
				]
			},

			"large_dataset": {
				"codes": [
					"BTCUSDT",
					"ETHUSDT",
					"BNBUSDT",
					"SOLUSDT",
					"ADAUSDT",
					"AVAXUSDT",
					"MATICUSDT",
					"UNIUSDT",
					"XRPUSDT",
					"ATOMUSDT"
				]
			},
			"extended_dataset": {
				"codes": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
							"LTCUSDT", "BCHUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT", "NEARUSDT", "AAVEUSDT", "FILUSDT", "SHIBUSDT", "ETCUSDT", "VETUSDT",
							"RUNEUSDT", "ALGOUSDT", "ICPUSDT", "LUNAUSDT", "EOSUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "1INCHUSDT"]
			},		
			"large_100_dataset": {
				"codes": [
					"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
					"LTCUSDT", "BCHUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT", "NEARUSDT", "AAVEUSDT", "FILUSDT", "SHIBUSDT", "ETCUSDT", 
					"VETUSDT", "RUNEUSDT", "ALGOUSDT", "ICPUSDT", "EOSUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "1INCHUSDT", "XLMUSDT",
					"TRXUSDT", "THETAUSDT", "FTMUSDT", "HBARUSDT", "XMRUSDT", "XTZUSDT", "EGLDUSDT", "FLOWUSDT", "KSMUSDT", "NEOUSDT",
					"CHZUSDT", "WAVESUSDT", "ZECUSDT", "ENJUSDT", "DASHUSDT", "YFIUSDT", "COMPUSDT", "BATUSDT", "RVNUSDT", "ZILUSDT",
					"ONEUSDT", "XEMUSDT", "HOTUSDT", "IOTAUSDT", "CELOUSDT", "ONTUSDT", "BTTUSDT", "QTUMUSDT", "ZRXUSDT", "OMGUSDT",
					"DGBUSDT", "ANKRUSDT", "CRVUSDT", "MINAUSDT", "KAVAUSDT", "ICXUSDT", "SRMUSDT", "GRTUSDT", "STORJUSDT", "ARUSDT",
					"IOTXUSDT", "SXPUSDT", "SCUSDT", "CELRUSDT", "SKLUSDT", "STMXUSDT", "SNXUSDT", "RSRUSDT", "OCEANUSDT", "RENUSDT",
					"ALPHAUSDT", "LRCUSDT", "KNCUSDT", "BNTUSDT", "REEFUSDT", "CTKUSDT", "BANDUSDT", "NKNUSDT", "FETUSDT", "DENTUSDT",
					"OGNUSDT", "MKRUSDT", "AUDIOUSDT", "DODOUSDT", "CAKEUSDT", "ROSEBUSDT", "CTSIUSDT", "ALICEUSDT", "GTUSDT", "ARPAUSDT",
					"LUNCUSDT", "BNXUSDT", "C98USDT", "APTUSDT", "OPUSDT", "GMTUSDT", "GALUSDT", "ENSUSDT", "FLMUSDT", "JASMYUSDT"
				]
			},   	
			"full_dataset": { 
				"codes": "*"
			},
			"active_dataset": "extended_dataset",  # Use medium dataset for better generalization
			"basis": "4h",
			"train_proportion": 0.7,
			"val_proportion": 0.15,
			"max_val_size": 12000,  # Maximum samples in validation set
			"max_test_size": 12000,  # Maximum samples in test set		
		}
	}  

 
	# Apply any overrides and create directories
	if override:
		apply_override(config, override)
		run_id = override.get('run_id', run_id)
		
	run_dir = Path(base_path) / run_id

	# Copy codes from the active_dataset into the main dataset_params entry
	updated_dataset_params = config['dataset_params'].copy()
	updated_dataset_params['codes'] = config['dataset_params'][config['dataset_params']['active_dataset']]['codes']

	config.update({
		'dataset_params': updated_dataset_params,
		'paths': {
			'run_dir': str(run_dir),
			'models_dir': str(run_dir / 'models'),
			'dataset_dir': '/mnt/hdd/streamlined_' + '-'.join(re.sub(r'USDT$', '', code) for code in updated_dataset_params['codes'])
		},
		'run_id': run_id
	})
	
	# Create directories
	run_dir.mkdir(parents=True, exist_ok=True)
	
	# Save configuration
	with open(run_dir / 'config.json', 'w') as f:
		json.dump(config, f, indent=4)
	
	return run_dir