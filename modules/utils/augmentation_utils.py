"""Contains configuration data for augmentaions"""
from torchvision import transforms

width_resize = 384
height_resize = 216

augmentation_options = {
    'vflip': transforms.RandomVerticalFlip(0.4),
    'hflip': transforms.RandomHorizontalFlip(0.4),
    'colorjitter': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
    'rot90': transforms.RandomRotation(90,expand=True),
    'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
    'contrast': transforms.RandomAutocontrast(p=0.5),
    'resize': transforms.Resize((height_resize, width_resize)),

    'randomaffine': transforms.RandomAffine(
        15, #degrees
        translate=(0.1, 0.1), #maximum absolute fraction horizontally and vertically
        scale=(0.8,1.5)) #scaling factor interval

}

rendezvous = ['vflip', 'hflip', 'colorjitter', 'rot90', 'resize']
tecno = ['randomaffine']

augmentation_prob = 0.7