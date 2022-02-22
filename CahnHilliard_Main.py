import sys
import numpy as np

Case     = str(int(float(sys.argv[1])))
gridSize = int(float(sys.argv[2]))
b        = float(sys.argv[3])
lmbda    = float(sys.argv[4])
seed_val = float(sys.argv[5])

# Cahn Hilliard simulations

import random
from dolfin import *
import io
import matplotlib
matplotlib.use('Agg') #For qsub batch file
import matplotlib.pyplot as plt

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(seed_val)
        super().__init__(**kwargs)
    def eval(self, values, x):
        if Case == str(1):
          values[0] = 0.5 + (random.uniform(-0.05, 0.05))
        elif Case == str(2):
          values[0] = 0.63 + (random.uniform(-0.05, 0.05))
        elif Case == str(3):
          values[0] = 0.75 + (random.uniform(-0.05, 0.05))
        else:
          print("A case number was not specified!")       
        values[1] = 0.0
    def value_shape(self):
        return (2,)


# Class for interfacing with the Newton solver
class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

#Convert image to grayscale
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#Save image as numpy array
def get_img_from_fig(fig):
    buf = io.BytesIO()
    extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(buf, format="rgba", dpi='figure',bbox_inches = extent,transparent = True, pad_inches = 0, cmap = "gray")
    buf.seek(0)
    img_arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                   newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    img_arr = rgb2gray (img_arr)
    print(img_arr.shape)
    buf.close()
       
    return img_arr

# Define constant model parameters:

# Model parameters
dt     = 5.0e-06  # time step
theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
# M = 1 (but M is neglected in the equations) 

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

#Create grid for initial conditions
mesh0 = UnitSquareMesh(gridSize, gridSize)
P0 = FiniteElement("Lagrange", mesh0.ufl_cell(), 1)
ME0 = FunctionSpace(mesh0, P0*P0)

# Create mesh and build function space
mesh = UnitSquareMesh(300, 300)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1*P1)

# Define trial and test functions of the space ``ME``:

# Define trial and test functions
du    = TrialFunction(ME)
q, v  = TestFunctions(ME)

# Define functions
uG  = Function(ME0)  # Initial condition on grid
u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step

# Split mixed functions
dc, dmu = split(du)
c,  mu  = split(u)
c0, mu0 = split(u0)

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
uG.interpolate(u_init)
u.interpolate(uG)
u0.interpolate(uG)

# Compute the chemical potential df/dc
c = variable(c)
f    = b*c**2*(1-c)**2
dfdc = diff(f, c)

# mu_(n+theta)
mu_mid = (1.0-theta)*mu0 + theta*mu

# Weak statement of the equations
L0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
L1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
L = L0 + L1

# Compute directional derivative about u in the direction of du (Jacobian)
a = derivative(L, u, du)

# Create nonlinear problem and Newton solver
problem = CahnHilliardEquation(a, L)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# Output
                     
# Step in time
t = 0.0
T = 5000*dt
i = 0 #counter
imglist = []

while (t < T):
    t += dt
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())
    cvalue = u.vector().get_local()[0::2]

    perc80 = np.sort(cvalue)[int(0.8*len(cvalue))]
    perc20 = np.sort(cvalue)[int(0.2*len(cvalue))]
    
    if (perc80>=0.92) and (perc20<=0.46):    
        if (t<=5.05e-04):
            i += 1
            if (i==10):
                plt.set_cmap('gray')
                plt.tight_layout(pad=0)
                plt.transparent=True
                plt.gca().set_axis_off()
                plt.gca().get_xaxis().set_visible(False)  
                plt.gca().get_yaxis().set_visible(False)
                plt.subplots_adjust(top = 1, bottom = 0, right = 1 , left = 0, 
                    hspace = 0, wspace = 0)
                plot(u.split()[0])
                fig = plt.gcf()
                #DPI = fig.get_dpi() #Gets DPI of image as plotted
                DPI = 500 # or manually set DPI
                fig.set_size_inches(2000/float(DPI),2000/float(DPI))
                plot_img_np = get_img_from_fig(fig)
                OneD_Img = plot_img_np.ravel() #Convert to 1D array
                imglist.append(OneD_Img)
                i = 0
            
        if (5.05e-04<t<=1.01e-03):
            i += 1
            if (i==20):
                plt.set_cmap('gray')
                plt.tight_layout(pad=0)
                plt.transparent=True
                plt.gca().set_axis_off()
                plt.gca().get_xaxis().set_visible(False)  
                plt.gca().get_yaxis().set_visible(False)
                plt.subplots_adjust(top = 1, bottom = 0, right = 1 , left = 0, 
                    hspace = 0, wspace = 0)
                plot(u.split()[0])
                fig = plt.gcf()
                #DPI = fig.get_dpi() #Gets DPI of image as plotted
                DPI = 500 # or manually set DPI
                fig.set_size_inches(2000/float(DPI),2000/float(DPI))
                plot_img_np = get_img_from_fig(fig)
                OneD_Img = plot_img_np.ravel() #Convert to 1D array
                imglist.append(OneD_Img)
                i = 0
            
        if (1.01e-03<t<=5.01e-03):
            i += 1
            if (i==50):
                plt.set_cmap('gray')
                plt.tight_layout(pad=0)
                plt.transparent=True
                plt.gca().set_axis_off()
                plt.gca().get_xaxis().set_visible(False)  
                plt.gca().get_yaxis().set_visible(False)
                plt.subplots_adjust(top = 1, bottom = 0, right = 1 , left = 0, 
                    hspace = 0, wspace = 0)
                plot(u.split()[0])
                fig = plt.gcf()
                #DPI = fig.get_dpi() #Gets DPI of image as plotted
                DPI = 500 # or manually set DPI
                fig.set_size_inches(2000/float(DPI),2000/float(DPI))
                plot_img_np = get_img_from_fig(fig)
                OneD_Img = plot_img_np.ravel() #Convert to 1D array
                imglist.append(OneD_Img)
                i = 0

        if (5.01e-03<t<=1.001e-02):
            i += 1
            if (i==100):
                plt.set_cmap('gray')
                plt.tight_layout(pad=0)
                plt.transparent=True
                plt.gca().set_axis_off()
                plt.gca().get_xaxis().set_visible(False)  
                plt.gca().get_yaxis().set_visible(False)
                plt.subplots_adjust(top = 1, bottom = 0, right = 1 , left = 0, 
                    hspace = 0, wspace = 0)
                plot(u.split()[0])
                fig = plt.gcf()
                #DPI = fig.get_dpi() #Gets DPI of image as plotted
                DPI = 500 # or manually set DPI
                fig.set_size_inches(2000/float(DPI),2000/float(DPI))
                plot_img_np = get_img_from_fig(fig)
                OneD_Img = plot_img_np.ravel() #Convert to 1D array
                imglist.append(OneD_Img)
                i = 0
            
        if (t>1.001e-02):
            i += 1
            if (i==200):
                plt.set_cmap('gray')
                plt.tight_layout(pad=0)
                plt.transparent=True
                plt.gca().set_axis_off()
                plt.gca().get_xaxis().set_visible(False)  
                plt.gca().get_yaxis().set_visible(False)
                plt.subplots_adjust(top = 1, bottom = 0, right = 1 , left = 0, 
                    hspace = 0, wspace = 0)
                plot(u.split()[0])
                fig = plt.gcf()
                #DPI = fig.get_dpi() #Gets DPI of image as plotted
                DPI = 500 # or manually set DPI
                fig.set_size_inches(2000/float(DPI),2000/float(DPI))
                plot_img_np = get_img_from_fig(fig)
                OneD_Img = plot_img_np.ravel() #Convert to 1D array
                imglist.append(OneD_Img)
                i = 0           

# Save numpy array into text file 
filename = "C{0}_b{1}_L{2}_G{3}_S{4}_savedImages.npy".format(Case,b,lmbda,gridSize,seed_val)          
np.savetxt(filename, np.array(imglist),fmt='%.5e')
