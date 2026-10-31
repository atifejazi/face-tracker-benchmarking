#!/usr/bin/env python3
"""Compare native 256px renders to their actual training targets, including temporal residuals."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser()
p.add_argument('--models',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
results={}
for model in a.models:
 root=model/'train/ours_25000'
 paths=sorted((root/'renders').glob('*.png'))
 psnr=[];temp=[];last_error=None
 for path in paths:
  pred=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255
  gt=np.asarray(Image.open(root/'gt'/path.name).convert('RGB'),dtype=np.float32)/255
  assert pred.shape==gt.shape
  error=pred-gt
  psnr.append(float(-10*np.log10(np.mean(error**2))))
  if last_error is not None:temp.append(float(np.mean(np.abs(error-last_error))))
  last_error=error
 assert len(paths)==95
 results[model.name]={'frames':len(paths),'psnr_mean_native_gt':float(np.mean(psnr)),
  'psnr_min_native_gt':min(psnr),'worst_frame':paths[int(np.argmin(psnr))].name,
  'temporal_error_mae_mean':float(np.mean(temp)),'temporal_error_mae_max':max(temp),
  'definition':'Temporal error = mean abs((pred_t-GT_t)-(pred_prev-GT_prev)), RGB [0,1]; preserves true image motion.'}
a.output.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
