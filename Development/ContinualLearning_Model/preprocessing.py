from torchvision import transforms
# add as many transforms as you like
transforms_1 = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.RandomPosterize(8, p=0.5),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


transforms_2 = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

transforms_3 = {  #the augmentation method AugMix
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

transforms_4 = { #compensate the AugMix() with the basic augmentation methods which are not contained
                 #in the AugMix()
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAutocontrast(),
        transforms.ColorJitter(),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

transforms_0 = {   #no augmentation operations
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

transforms_5 = {    #concatenate two augmentation methods RandAugment() and TrivialAugmentWide()
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

transforms_6 = {   #apply the single basic augmentation operation
                   #among which the HorizaontalFlip and VerticalFlip have gaiend
                   #improvement
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.Grayscale(num_output_channels=3),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        ###transforms.RandomRotation(degrees=(0, 180)),
        ###transforms.RandomInvert(),
        #transforms.RandomPosterize(bits=2),
        #transforms.RandomSolarize(threshold=192.0),
        #transforms.RandomAdjustSharpness(sharpness_factor=2),
        #transforms.RandomAutocontrast(),
        #?????AutoAugmentPolicy.IMAGENET
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomEqualize(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=96),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])
img_transforms = transforms.Compose([transforms.RandomResizedCrop(size=96),
                                    transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])