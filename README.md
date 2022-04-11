# colorization

Neural Network model built for the computer vision class final project, developed using the PyTorch library and with the help of the architechture proposed by Zhang in his 2016 paper: [Zhang](http://arxiv.org/pdf/1603.08511.pdf)

- The model uses a series of 2d convolutional layers in order to convert a grayscale input image (formaly known as the luminance channel) into its respective colorized prediction
- Instead of using the widespread RGB colour space, the Lab space has been chosen in order to reduce the amount of channels to predict and the degree of quantization such colour space can have as it is represented by cartesian coordinates, depicting levels of chroma in the different spectres (blue, red, yellow and green)
- Trained with 5000 image files with a 400x400 pixel format.


Training will be performed on plain CPU, looking forward to improve result using cuda
