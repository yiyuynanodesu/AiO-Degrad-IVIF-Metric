# All in one degradation infrared and visible image fusion

## Environment

```python
# first you need to prepare clip-iqa enviroment, see Metric
pip install scipy
pip install scikit-learn
pip install openpyxl
pip install numpy==1.23.5
pip install Pillow==9.5
```

## Usage

use this template or see `Metric/metric_template.py`

```python
from evaluate.eval_metric import eval_batch as normal_eval
from evaluate.degrad_eval_metric import eval_batch as degrad_eval
from evaluate.degrad_eval_metric_by_type import eval_batch as degrad_type_eval

import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training Example')
parser.add_argument('--model', default = '', help='model')
parser.add_argument('--dataset', default = '', help='dataset')
parser.add_argument('--ir_path', default = '', help='ir path')
parser.add_argument('--vi_path', default = '', help='vi path')
parser.add_argument('--result_path', default = '', help='result path')
parser.add_argument('--save_path', default = './', help='model')
args = parser.parse_args()

# Degrad IVIF
if args.dataset == 'DDL':
    # DDL
    degrad_eval(args.result_path, args.save_path, model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_degrad.xlsx')
    # Degrad Type IVIF
    degrad_type_eval(args.result_path, args.save_path, degard_list=['HazeRain','HazeLow','Rain','Haze','Exposure','Light'], model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_detail.xlsx')
elif args.dataset == 'EMS':
    degrad_eval(args.result_path, args.save_path, model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_degrad.xlsx')
    degrad_type_eval(args.result_path, args.save_path, degard_list=['Rain','Haze','Exposure','Light'], model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_detail.xlsx')
else:
    # Normal IVIF
    normal_eval(args.ir_path, args.vi_path, args.result_path, args.save_path, model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_normal.xlsx')
```

or

```python
python metric_template.py --dataset "xxxx" --model "xxxx" --result_path "xxxx" --vi_path "xxxx" --ir_path "xxxx"
```

## Warning List
### SPGFusion
if you use SPGFusion, add this to eval_metric.py's line 36:
```python
w_vi, h_vi = vi_img.size  # (width, height)
w_ir, h_ir = ir_img.size
new_w = max(16, (w_vi // 16) * 16)
new_h = max(16, (h_vi // 16) * 16)
if (w_vi != new_w) or (h_vi != new_h):
    vi_img = vi_img.resize((new_w, new_h), resample=Image.BICUBIC)
if (w_ir != new_w) or (h_ir != new_h):
    ir_img = ir_img.resize((new_w, new_h), resample=Image.BICUBIC)
```

## Model Flops and Params/Speed
```python
import torch.nn as nn
class Network(nn.Module):
    def __init__(self, model, model_pretrain=None):
        super().__init__()
        self.model = model
        self.model_pretrain = model_pretrain
    def forward(self, a, b, text=None):
        if self.model_pretrain != None:
            ### use your pretrain model to do sth
            pass
        ### real model fusion
        output = self.model(a,b) 
        return output

net = Network(your_real_model)
### Create ir, vi input tensor
from thop import profile
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
#### model fusion ####
flops, _ = profile(net, inputs=(vi, ir))
total = sum([params.nelement() for params in net.parameters()])
#### model fusion ####
start.record()
output = net(vi, ir)
end.record()
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)
flops_g = flops / 1e9  # to GFLOPs
params_m = total / 1e6  # to MParams
speed_s = elapsed_time / 1000  # to Second
print(f'GFLOPs: {flops_g:.2f}, MParams: {params_m:.2f}, Speed: {speed_s:.3f}s')

### if you meet nn.DataParallel you can use this:
net = Network(your_real_model)
### Create ir, vi input tensor
from thop import profile
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
#### model fusion ####
start.record()
output = net(vi, ir)
end.record()
#### model fusion ####
vi_profile = vi.unsqueeze(0)
ir_profile = ir.unsqueeze(0)
flops, _ = profile(net, inputs=(vi_profile, ir_profile))
total = sum([params.nelement() for params in net.parameters()])
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)
flops_g = flops / 1e9  # to GFLOPs
params_m = total / 1e6  # to MParams
speed_s = elapsed_time / 1000  # to Second
print(f'GFLOPs: {flops_g:.2f}, MParams: {params_m:.2f}, Speed: {speed_s:.3f}s')
```

## Acknowledgement

Our code is based on the following excellent work:

[LLVIP: A Visible-infrared Paired Dataset for Low-light Vision](https://github.com/bupt-ai-cz/LLVIP)

[Exploring CLIP for Assessing the Look and Feel of Images (AAAI 2023)](https://github.com/IceClear/CLIP-IQA/blob/v1-3.6/demo/clipiqa_single_image_demo.py)

[RollingPlain/IVIF_ZOO: Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption. A fire-new survey for infrared and visible image fusion.](https://github.com/RollingPlain/IVIF_ZOO)

[No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency (WACV 2022) ](https://github.com/isalirezag/TReS)

[Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation](https://github.com/JinyuanLiu-CV/SegMiF)

[DeepSeek(mainly contributor for Tool)](https://www.deepseek.com/)