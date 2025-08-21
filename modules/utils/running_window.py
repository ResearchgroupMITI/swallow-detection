import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

# Load file
manometry_file = '../data/sensors/manometry_5.pkl'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#manometry = pd.read_csv(manometry_file, sep=',')
manometry = pd.read_pickle(manometry_file)

# Apply running mean to smooth the sensor signals
window_size = 30
manometry = manometry.rolling(window=window_size, center=True).mean()
manometry = np.array(manometry.fillna(0))

# Clip and normalize values from 0 to 255
manometry = manometry.clip(-200, 300)
manometry = (255*(manometry - np.min(manometry))/np.ptp(manometry)).astype(int)
manometryT = torch.Tensor(manometry).to(device=device)

window_size = 700
step_size = 1


for i in range(0, manometry.shape[0] - window_size, step_size):
    # get window i
    manometry_window_i = manometryT[i:i+window_size, :].T

    # add dimension
    manometry_window_i = manometry_window_i.unsqueeze(0)

    # resize Tensor 224x224
    resized_tensor = TF.resize(manometry_window_i, size=[224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
    resized_tensor = resized_tensor.squeeze(0)

    # alternative: interpolate Tensor 224x224
    # resized_tensor_bilinear = F.interpolate(manometry_window_i.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    
    # to plot: Tensor stacking and to cpu and numpy
    # resized_array = resized_tensor.unsqueeze(0)
    # resized_array = torch.stack([resized_array] * 3, dim=1)
    # resized_array = resized_array.squeeze(0).cpu().numpy().astype(np.uint8)

    # apply color map
    #colormapped_image = cv2.applyColorMap(resized_array, cv2.COLORMAP_JET)

    # save images
    # cv2.imwrite('output/moving_window.jpg', colormapped_image)
