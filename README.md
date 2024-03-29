# Mechanical MNIST Cahn-Hilliard Dataset

The Mechanical MNIST Cahn-Hilliard dataset is an extension of the [Mechanical-MNIST](https://github.com/elejeune11/Mechanical-MNIST) dataset collection. 
This repository contains the codes used to create this dataset as implemented in the paper ''Enhancing Mechanical Metamodels with a Generative Model-Based Augmented Training Dataset'' [access paper](https://arxiv.org/abs/2203.04183).  

The input domain distributions are generated from a Finite Element implementation of the Cahn-Hilliard equation by varying four simulation parameters: the initial concentration, the grid size on which the concentration is initialized, the parameter lambda related to the interface thickness, and b, the peak-to-valley value of the symmetric double-well chemical free-energy function. The dataset consists of 104,813 different stripe and circle patterns saved as 400 x 400 binary bitmap images each in a separate text file. The binary images are converted into two-dimensional meshed domains using the OpenCV library, Pygmsh, and Gmsh 4.6.0. We also include a downsampled version of the patterns stored in a single text file where each row contains 64 x 64 = 4,096 entries.

We perform Finite Element simulations of the heterogeneous meshed domains modelled as unit squares of Neo-Hookean binary material subject to large equibiaxial extension deformation with fixed displacements **d** = [0.0,0.001,0.1,0.2,0.3,0.4,0.5]. We provide the simulation results consisting of the following: 

1. change in strain energy reported at each level of applied displacement
2. total reaction force at the four boundaries reported at each level of applied displacement
3. full field displacement reported at the final applied displacement **d**=0.5

All Finite Element simulations are conducted with the [FEniCS](https://fenicsproject.org) computing platform.

# Full Dataset

The full dataset is hosted by OpenBU at [Link to dataset](https://hdl.handle.net/2144/43971)

![Github_pic](https://user-images.githubusercontent.com/77171178/155850904-c11710a8-6e33-4a8b-8dc9-d5688e9ba1a9.png)

## In this Repository
This repository contains the codes used to generate the dataset.
* [`CahnHilliard_Main.py`](CahnHilliard_Main.py) -- Code to generate the Cahn-Hilliard patterns. The inputs to the code are passed as command line arguments in the order Case, Grid Size, b, lambda, and the value of the random seed.
As an example, you can run the code using the following command with the FEniCS module loaded:
```
python3 -i CahnHilliard_Main.py 1 97 8.43557e+01 1.81208e-02 95
```  

* [`NumpyImageToGmsh.py`](NumpyImageToGmsh.py) -- Code to generate _.xdmf_ mesh files based on Cahn-Hilliard patterns. The `example` folder provides more details on running the code and includes sample inputs and outputs.


* [`Equibiaxial_Hyperelastic.py`](Equibiaxial_Hyperelastic.py) -- Code to generate the FEA simulation results. The code takes a single input, the meshed domain saved in _.xdmf_ format, as a command line argument. 
As an example, you can run the code using the following command with the FEniCS module loaded:
```
python3 -i Equibiaxial_Hyperelastic.py Image2931.xdmf
```

* [`Using_Csv_CH_Database.ipynb`](Using_Csv_CH_Database.ipynb) -- A Jupyter notebook demonstrating how to navigate through the provided _.csv_ database to find desired simualtaion parameters of a certain image pattern or a Cahn-Hilliard simulation and locate the results in the corresponding _.txt_ files.
