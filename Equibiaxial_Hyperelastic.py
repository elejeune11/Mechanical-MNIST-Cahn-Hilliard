from dolfin import *
from fenics import *
from ufl import nabla_grad
from ufl import nabla_div
import numpy as np
import sys
import os
import os.path

flag_quad = 2

# Folder
folder_name =  '/projectnb2/lejlab2/Hiba/Equi_Hyper/Results'
if not os.path.exists(folder_name):
    os.makedirs(folder_name, exist_ok=True)

# Import mesh
meshfile     = sys.argv[1]

# Extract mesh file name without extension
meshfileName = os.path.basename(meshfile)
meshName = meshfileName[0:len(meshfileName)-5]

# Read mesh file
mesh = Mesh()
with XDMFFile(meshfile) as infile:
    infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(meshfile) as infile:
    infile.read(mvc,"SurfaceRegions")
    sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)
dx = Measure('dx', subdomain_data=sub, domain=mesh)

# Class to define different material properties in subdomains
class EMod(UserExpression):
    def __init__(self,subdomains,E_list,**kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.E_list = E_list
    def value_shape(self):
        return () 
    def eval_cell(self,values,x,cell):
        values[0] = 0
        for i in range(len(E_List)):
                if self.subdomains[cell.index] == 1:
                  values[0] = self.E_list[0]
                else: 
                  values[0] = self.E_list[1]

# Define material properties
E_List = [1 , 10]
nu = Constant(0.3)
E = EMod(sub,E_List,degree=0)
V0 = FunctionSpace(mesh,'DG',0)
E = project(E,V0)

mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

# Compliler settings / optimization options 
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = flag_quad

# Define function space to solve the problem
P2 = VectorElement("Lagrange", mesh.ufl_cell(), flag_quad)
TH = P2
W = FunctionSpace(mesh, TH)

# Define traction on the boundary and body forces
T  = Constant((0.0, 0.0)) 
B  = Constant((0.0, 0.0))

# Define finite element problem
u = Function(W)
du = TrialFunction(W)
v = TestFunction(W)

# Define problem to be solved for different displacement at the boundary
def problem_solve(applied_disp,u,du,v):

    left = CompiledSubDomain('on_boundary && near(x[0], 0, tol)', tol=1E-5)
    right = CompiledSubDomain('on_boundary && near(x[0], 1, tol)', tol=1E-5)
    btm = CompiledSubDomain('on_boundary && near(x[1], 0, tol)', tol=1E-5)
    top = CompiledSubDomain('on_boundary && near(x[1], 1, tol)', tol=1E-5) 
      
    # Updated boundary conditions 
    lftBC = DirichletBC(W.sub(0), Constant((-1.0*applied_disp/2.0)), left)
    rgtBC = DirichletBC(W.sub(0), Constant((applied_disp/2.0)), right)
    topBC = DirichletBC(W.sub(1), Constant((applied_disp/2.0)), top)
    btmBC = DirichletBC(W.sub(1), Constant((-1.0*applied_disp/2.0)), btm)    
      
    bcs = [lftBC, rgtBC, topBC, btmBC] 
    
    # Kinematics
    d = len(u)
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    F = variable(F)

    psi = 1/2*mu*( inner(F,F) - 3 - 2*ln(det(F)) ) + 1/2*lmbda*(1/2*(det(F)**2 - 1) - ln(det(F)))
    f_int = derivative(psi*dx,u,v)
    f_ext = derivative( dot(B, u)*dx('everywhere') + dot(T, u)*ds , u, v)
    Fboth = f_int - f_ext 
    # Tangent 
    dF = derivative(Fboth, u, du)
    solve(Fboth == 0, u, bcs, J=dF)

    P = diff(psi,F) 
    S = inv(F)*P  
    sig = F*S*F.T*((1/det(F))*I)
      
    return u, du, v, f_int, f_ext, psi 
    
to_print = True

def rxn_forces(list_rxn,W,f_int,f_ext):
    x_dofs = W.sub(0).dofmap().dofs()
    y_dofs = W.sub(1).dofmap().dofs()
    f_ext_known = assemble(f_ext)
    f_ext_unknown = assemble(f_int) - f_ext_known
    dof_coords = W.tabulate_dof_coordinates().reshape((-1, 2))
    y_val_min = np.min(dof_coords[:,1]) + 10E-5; y_val_max = np.max(dof_coords[:,1]) - 10E-5
    x_val_min = np.min(dof_coords[:,0]) + 10E-5; x_val_max = np.max(dof_coords[:,0]) - 10E-5
    # Reaction forces in the y-direction: top and bottom boundaries
    y_top = []; y_btm = [] 
    for kk in y_dofs:
        if dof_coords[kk,1] > y_val_max:
            y_top.append(kk)
        if dof_coords[kk,1] < y_val_min:
            y_btm.append(kk)
    f_sum_top_y = np.sum(f_ext_unknown[y_top])
    f_sum_btm_y = np.sum(f_ext_unknown[y_btm])    
    # Reaction forces in the x-direction: right and left boundaries
    x_lft = []; x_rgt = [] 
    for kk in x_dofs:
        if dof_coords[kk,0] > x_val_max:
            x_rgt.append(kk)
        if dof_coords[kk,0] < x_val_min:
            x_lft.append(kk)
    f_sum_rgt_x = np.sum(f_ext_unknown[x_rgt])
    f_sum_lft_x = np.sum(f_ext_unknown[x_lft])
    
    if to_print: 
        print("y_top, y_btm rxn force:", f_sum_top_y, f_sum_btm_y)
        print("x_lft, x_rgt rxn force:", f_sum_lft_x, f_sum_rgt_x)
    
    # Check that forces sum to 0 
    sum_x =  f_sum_lft_x + f_sum_rgt_x
    sum_y =  f_sum_top_y + f_sum_btm_y 
    
    if to_print:
        print('sum forces x:',sum_x )
        print('sum forces y:',sum_y )
        
    list_rxn.append([f_sum_lft_x,f_sum_rgt_x,f_sum_top_y,f_sum_btm_y])
    return list_rxn

# Displacements on a 64 x 64 grid
def pix_centers(u):
    disps_all_x = np.zeros((64,64))
    disps_all_y = np.zeros((64,64))
 
    midX = np.linspace(0,1,65)+0.5/64 
    midY = np.linspace(0,1,65)+0.5/64
      
    for kk in range(0,64):
          for jj in range(0,64):
              disps_all_x[jj,kk] = u(midX[kk],midY[jj])[0]
              disps_all_y[jj,kk] = u(midX[kk],midY[jj])[1]
      
    return disps_all_x, disps_all_y

# Get strain energy
def strain_energy(list_psi, psi):
    val = assemble(psi*dx)
    list_psi.append(val)
    return list_psi

# Get delta strain energy
def strain_energy_subtract_first(list_psi):
    first = list_psi[0]
    for kk in range(0,len(list_psi)):
        list_psi[kk] = list_psi[kk] - first 
    return list_psi

# Set up file name for paraview (if needed)
fname_paraview = File(folder_name +'/'+ meshName + "_paraview.pvd")

# Initialize empty lists for reaction forces and delta potential energy
list_rxn = []

list_psi = [] 

# Solve for the list of displacements 
disp_val = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]

fname = folder_name + '/'+ meshName +'_pixel_disp' 

for dd in range(0,len(disp_val)):
    applied_disp = disp_val[dd]
    u, du, v, f_int, f_ext, psi = problem_solve(applied_disp,u,du,v)
    list_rxn = rxn_forces(list_rxn,W,f_int,f_ext)
    #fname_paraview << (u,dd)
    disps_all_x, disps_all_y = pix_centers(u)
    fn_x = fname + '_' + str(applied_disp) + '_x.txt'
    fn_y = fname + '_' + str(applied_disp) + '_y.txt'
    np.savetxt(fn_x,disps_all_x,fmt='%.5e')
    np.savetxt(fn_y,disps_all_y,fmt='%.5e')
    list_psi = strain_energy(list_psi, psi)

# Save reaction forces
fname = folder_name +'/'+ meshName +'_rxn_force.txt'
np.savetxt(fname,np.asarray(list_rxn),fmt='%.5e')

# Save delta (total) potential energy 
fname = folder_name +'/'+ meshName +'_strain_energy.txt'
list_psi = strain_energy_subtract_first(list_psi)
np.savetxt(fname, np.asarray(list_psi),fmt='%.5e')
