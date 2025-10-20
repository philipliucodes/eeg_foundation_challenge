# EEG Foundation Challenge 1

Code for EEG Foundation Challenge 1.  

- `ccd_windows.py`: loads and prepares the CCD dataset, building subject-wise train/valid/test splits.  
- `braindecode.py`: trains a regression model using any architecture from the Braindecode library.  

Run training with:  

<pre> ```bash python braindecode.py --cache_root /path/to/cache [other arguments...] ``` </pre>

## Arguments

### Required
--cache_root (str, required)  
Path to the CCD cache directory. Example: /data/ccd  

### Data
--mini (flag)  
Use a small subset for quick smoke tests.  

### Target & Loss
--target_space {rt, logrt} (default: rt)  
Train on raw RT (“rt”) or log-RT (“logrt”). When “logrt”, training targets are clamped to [rt_min, rt_max] and evaluation is exponentiated back to RT.  
--loss {huber, mse} (default: huber)  
Regression loss.  
--huber_delta (float, default: 0.05)  
δ parameter for Huber when using huber loss.  
--rt_min (float, default: 0.2), --rt_max (float, default: 2.0)  
Clamp range for RT when using logRT.  

### Optimization
--batch_size (int, default: 128)  
Mini-batch size.  
--epochs (int, default: 100)  
Maximum training epochs.  
--lr (float, default: 1e-3)  
Learning rate for AdamW.  
--weight_decay (float, default: 1e-5)  
Weight decay for AdamW.  
--num_workers (int, default: 2)  
Number of DataLoader workers.  
--arch (choice, default: eegnex)  
Model architecture (see ARCH_CHOICES in the file).  

### Misc & Run Settings
--patience (int, default: 15)  
Early-stopping patience on validation nRMSE.  
--min_delta (float, default: 0.0)  
Minimum improvement in nRMSE to reset patience.  
--save_path (str, default: <arch>.pt)  
Where to save best model weights.  
--seed (int, default: -1)  
Random seed. If <0, a secure auto-seed is used.  
--deterministic (flag)  
Enable deterministic PyTorch/CuDNN (slower, reproducible).  

## Notes
- Data is windowed and split by subject using `ccd_windows.py`.  
- When target_space=logrt the model trains in log-space, but metrics are reported in RT units.  
