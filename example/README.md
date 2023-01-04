# example

Here we provide a brief tutorial on how to use the [`NumpyImageToGmsh.py`](NumpyImageToGmsh.py) code to ultimately generate the _.xdmf_ mesh files needed to run the provided FEA code.

* The [`input_patterns`](input_patterns) folder contains 6 pattern examples from the Cahn-Hilliard pattern dataset. 

* The [`mesh_files`](mesh_files) folder contains the output of the [`NumpyImageToGmsh.py`](NumpyImageToGmsh.py) code (_.py_ files) as well as the 3 output files (_.msh_, _.xdmf_, and _.h5_ files) of subsequently running the resulting _.py_ files in `Gmsh 4.6.0`.

The [`NumpyImageToGmsh.py`](NumpyImageToGmsh.py) code expects that the Cahn-Hilliard patterns are stored in [`input_patterns`] folder. To run the code, simply use the following command with python3 loaded.

'''
python3 NumpyImageToGmsh.py
'''

The output of this code are python files to be run in `Gmsh 4.6.0`. To obtain the _.xdmf_ mesh files, _Image_#_.py_
