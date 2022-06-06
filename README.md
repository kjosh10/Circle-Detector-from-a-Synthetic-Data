# Circle Detector using Deep Neural Networks on a Synthetic Data

The project focuses on creating a synthetic data comprising of various shapes, viz. Circle, Rectangle, Triangle and Ellipses.
After the generation of synthetic data a comparison of Feed-forward network is done with Convolution Neural Networks (CNN).

## Aim of the project

The main aim of the project is to compare Feed-forward network with CNN's. It was realized during the project that the differentiation between ellipse and the circle was almost impossible when Feed-forward network was used.
Their was no learning observed when implementing Feed-forward network for differntiating between Circles and Ellipses.
On the contrary, the CNN's were still learning and able to differentiate between a circle and an ellipse. be solved and visualized using the program.

This project files can be explained as below:- 

### \util\
The above folder comprises of the various utilies used to generate Synthetic data, model and visualize the model summary.

### \Images
The above folder comprises of the Images generated using the customized synthetic_data_generator utility.

### \circle_detector
The above jupyter notebook comprises of implementing all the utilities and visualzing the project.

### \config
The above file consists of all the user-defined parameters required for the project. 

### \requirements.txt
The above file can be conviniently used to create a virtual environment in the personal Computers/Laptops.