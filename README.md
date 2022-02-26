# Mechanical MNIST Cahn-Hilliard Dataset

The Mechanical MNIST Cahn-Hilliard dataset is an extension of the [Mechanical-MNIST](https://github.com/elejeune11/Mechanical-MNIST) dataset collection. 

The input domain distributions are generated from a Finite Element implementation of the Cahn-Hilliard equation by varying four simulation parameters: the initial concentration, the grid size on which the concentration is initialized, the parameter lambda related to the interface thickness, and the peak-to-valley value of the symmetric double-well chemical free-energy function. The dataset consists of 104,813 different stripe and circle patterns saved as 400 x 400 binary bitmap images. The binary images are converted into two-dimensional meshed domains using the OpenCV library, Pygmsh, and Gmsh 4.6.0. 

We perform Finite Element simulations of the heterogeneous meshed domains modelled as unit squares of Neo-Hookean binary material subject to large equibiaxial extension deformation with fixed displacements d = [0.0,0.001,0.1,0.2,0.3,0.4,0.5]. We provide the simulation results consisting of the following: (1) change in strain energy reported at each level of applied displacement, (2) total reaction force at the four boundaries reported at each level of applied displacement, and (3) full field displacement reported at the final applied displacement d=0.5. All Finite Element simulations are conducted with the [FEniCS](https://fenicsproject.org) computing platform.

# Full Dataset

The full dataset is hosted by OpenBU at [**Link to dataset**]()

![Artboard 1]()

## In This Repository
This repository contains the codes used to generate the dataset.
* [`CahnHilliard_Main.py`](CahnHilliard_Main.py) -- Code to generate the Cahn-Hilliard patterns. The inputs to the code are passed as command line arguments in the order 
 
* Datasets ([`data.zip`](data.zip)): Compressed versions of the datasets used in this work for metamodel training and testing
* Jupyter Notebook ([`metamodel.ipynb`](metamodel.ipynb)): Tensorflow implementation of our metamodel in addition to a more detailed explanation on metamodel and generative model training and testing



