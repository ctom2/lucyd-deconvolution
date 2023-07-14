# LUCYD: A Feature-Driven Richardson-Lucy Deconvolution Network

Folder structure:

```
lucyd-deconvolution
│   lucyd.py
│   evaluate.py
│   train.py
│
└───utils
│   │   loader.py
│   │   ssim.py
│
└───data
│   │   gt
│   │   nuc
│   │   act
```

## Prerequisities:
* Python 3.7 or higher
* PyTorch 1.12.1 or higher

## Training:
```
model = LUCYD(num_res=1)
model = train(model, train_dataloader, test_dataloader)
```

## Testing:
```
evaluate(model, eval_dataloader)
```
