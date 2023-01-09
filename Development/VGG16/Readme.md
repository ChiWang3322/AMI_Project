## Data Preprocessing(Zheng Tao)

- Extract the image inside the bounding box

- Naming convention: box1.png(jpeg)

- Resize: resize all the box to same size, let's say the average size of the box

  

### Labeling and Data augmentation(Xue Zhang)

- Convert the label from dent, scratch to number 1,2,3,4

- Create a csv file, the first column is the image name(box1.png), the second column is it's label

- Augment data, rotate, flip, noise...

  

### Model preparation(Chi Wang)

- Implement or find available VGG model
- Implement validation method
- Implement these classes: Dataset, Dataloader, train, model...



## What we should deliver by the end of 27.06.2022

- A find tuned VGG model
- A well organized dataset including boxi.png(jpeg)
- Some new idea...(optional)

# New issues, 01.07.2022
- Feature augmentation(rim)
- How to train our network(different image size,from 20x20 to 220x220)
- Build a classifier for wheel
- Transfer learning, pretrain model on car dataset
- Autoencoder(Unet)

