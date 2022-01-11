# Attempt at implementing/reproducing: Density Estimation using Real NVP (ID2223 Project).
Link to paper: [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803)

Some code adopted from Chris Chute's repository: [Real NVP - Chris Chute](https://github.com/chrischute/real-nvp)


1. In order to run the code just open a terminal and type:
2. ``` python main.py ```
3. Note that before you run it, you need to open the main.py file and insert the save path.
4. If you don't have CUDA, you might want to remove the ".cuda()" method etc etc
5. Even if you DO have an Nvidia GPU, it might not support Automatic Mixed Precision (AMP). In that case, disable autocasting and associated Grad Scaler by setting:
6.  ``` use_amp = False ``` at the top of the  ``` main.py ``` file.

