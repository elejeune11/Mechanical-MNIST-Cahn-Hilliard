'''
Read image and convert into Gmsh code
'''

import numpy as np
import cv2
import os
import sys

# Directory to load files
images_folder = 'input_patterns/'

# Directory to save files
mesh_folder = 'mesh_files/'
if not os.path.exists(mesh_folder):
    os.mkdir(mesh_folder)

# Load image array
files = [file for file in os.listdir(images_folder)]

# Sort files
files_sort = sorted(files) 

# Count total number of images
total_img_nmbr = 1

for fl in files_sort:
    orig_img1 = np.loadtxt(os.path.join(images_folder, fl)) 

    #Extract file name without extension
    filename = fl[0:len(fl)-4] 
    
    # Size of image
    Img_size = orig_img1.shape[1]

    _, threshold = cv2.threshold(orig_img1, 110, 255, cv2.THRESH_BINARY_INV)
    
    threshold = threshold.astype(np.uint8)
    
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_TC89_KCOS)
     
    # To save original coordinates of vertices
    Orig_x_lst = []
    Orig_y_lst = []
    
    # To save coordinates of vertices
    x_lst = []
    y_lst= []
    approx = []
    
    # Reduce image size to a unit square
    reduced_Img_size = Img_size/Img_size
    y_mean = reduced_Img_size / 2
    
    # Get coordinates of patterns
    for cnt in contours :
      
        approx = cv2.approxPolyDP(cnt, 0.00009 * cv2.arcLength(cnt, True), True)
      
        # Draw boundary of contours
        cv2.drawContours(orig_img1, [approx], 0, (0, 0,0), 3)
     
        # Flatten the array containing the coordinates of the vertices
        n = approx.ravel()
        i = 0
    
        for j in n :
            if(i % 2 == 0):
                #Original coordinates
                x = n[i]
                y = n[i + 1]
                
                Orig_x_lst.append(x)
                Orig_y_lst.append(y)
    
            i += 1
    
    Max_x = max(Orig_x_lst)
    Max_y = max(Orig_y_lst)
    
    # Characteristic length
    cl = 0.01
    string_cl = repr(cl)
    
    # domain extent
    xmin, ymin   = 0 , 0
    xmax, ymax   = 1 , 1
    
    # Frame coordinates
    x_bottom = np.arange(xmin,xmax+cl,cl)
    y_bottom = (ymin)*np.ones(len(x_bottom))
    
    x_top = np.flip(x_bottom)
    y_top = (ymax)*np.ones(len(x_top))
    
    y_right = np.arange(ymin,ymax+cl,cl)
    x_right = (xmax)*np.ones(len(y_right))
    
    y_left = np.flip(y_right);
    x_left = (xmin)*np.ones(len(y_left))
    
    x_Frame = np.ravel([x_bottom,x_right,x_top,x_left])
    y_Frame = np.ravel([y_bottom,y_right,y_top,y_left])
    z_Coord = np.zeros(len(x_Frame))
    
    FrameCoord = np.column_stack((x_Frame,y_Frame,z_Coord))
    
    # Remove duplicate points
    index = np.unique(FrameCoord, axis = 0, return_index=True)[1]
    sorted_index = np.sort(index)
    FrameCoord = FrameCoord[sorted_index]
    
    # as string and disable truncation 
    np.set_printoptions(threshold=sys.maxsize, floatmode='maxprec_equal') 
    string_FrameCoord = np.array2string(FrameCoord, separator=',')
    
    
    # Write as Gmsh code
    filename_gmsh = mesh_folder + "{0}.py".format(filename)
    gmshFile = open(filename_gmsh,'w')
    
    # Headers
    inputList = ["import pygmsh \n", "import meshio \n","import numpy as np \n",
                 " \ngeom = pygmsh.opencascade.Geometry() \n \n"]
    gmshFile.writelines(inputList)
    
    # Mesh size
    inputList = ["# mesh size description \n","cl = " + string_cl + "\n \n"]
    gmshFile.writelines(inputList)
    
    # Frame, Background domain
    inputList = ["# Coordinates of Frame \n","FrameCoordinates = np.array("
                 + string_FrameCoord + ")\n \n",
                 "Dom0 = geom.add_polygon(FrameCoordinates,cl,make_surface=True);\n \n",
                 "# Define coordinates of Inner Domain \n"]
    gmshFile.writelines(inputList)
    
    # Extract coordinates of resized image
    # Print each inner domain coordinates into Gmsh 
    nb_cnt = 1 #count number of contours
    for cnt in contours :
        
        # Contour number as string
        string_cnt = repr(nb_cnt)
        nb_cnt += 1
        
        approx = cv2.approxPolyDP(cnt, 0.00009 * cv2.arcLength(cnt, True), True)
      
        # Draw boundary of contours
        cv2.drawContours(orig_img1, [approx], 0, (0, 0, 255), 3)
     
        # Flatten the array containing the coordinates of the vertices
        n = approx.ravel()
        
        l = 0
        x_lst.clear()
        y_lst.clear()
        for k in n:
    
            if(l % 2 == 0):
                #Resized image coordinates (unit square)
                x_red = n[l]/Max_x
                y_red = n[l + 1]/Max_y            
                
                # Reorient the image
                if y_red > y_mean:
                    y_reorient =  y_red - 2*abs(y_red-y_mean)
                elif y_red < y_mean:
                    y_reorient =  y_red + 2*abs(y_red-y_mean)
                else:
                    y_reorient = y_red
                    
                x_lst.append(x_red)
                y_lst.append(y_reorient)
    
            l += 1    
 
        # Contour coordinates for each contour (pattern domain)
        Domain_z_Coord = np.zeros(len(x_lst))
        DomainCoord = np.column_stack((x_lst,y_lst,Domain_z_Coord))
    
        # Remove duplicates from contour coordinates
        dom_index = np.unique(DomainCoord, axis = 0, return_index=True)[1]
        sorted_dom_index = np.sort(dom_index)
        DomainCoord = DomainCoord[sorted_dom_index]
       
        # As string
        string_DomainCoord = np.array2string(DomainCoord, separator=',')
       
        inputList = ["PolCoordinates" + string_cnt +"= np.array("
                 + string_DomainCoord + ")\n \n", "Dom" + string_cnt + "= geom.add_polygon(PolCoordinates" 
                 + string_cnt + ",cl,make_surface=True);\n \n"]
        gmshFile.writelines(inputList)
    
    # List domains for boolean operation
    
    # Check parent/child contour hierarchy
    is_parent = []
    is_child = []
    
    lst_contours = list(range(1,len(contours)+1))
    
    for i in range(len(contours)):
        if hierarchy[0,i,2] != -1:
            is_parent.append(i+1)
            is_child.append(int(hierarchy[0,i,2]+1))
            if (i+1) in lst_contours:
                lst_contours.remove(i+1)
            if (hierarchy[0,i,2]+1) in lst_contours:
                lst_contours.remove(int(hierarchy[0,i,2]+1))
    
            
    string_domains = ["Dom{0}".format(i) for i in (lst_contours)]
    string_domains_form = ("{0}".format(','.join(map(str,string_domains))))
    
    string_parents = ["Dom{0}".format(i) for i in (is_parent)]
    string_parents_form = ("[{0}]".format(','.join(map(str,string_parents))))
    
    string_children = ["Dom{0}".format(i) for i in (is_child)]
    string_children_form = ("[{0}]".format(','.join(map(str,string_children))))
    
    # Combine parents and children in one list, sort and remove duplicates
    parent_child_lst = list(dict.fromkeys(sorted(is_parent+is_child)))
    
    belong_to_surface1 = parent_child_lst[1:][::2] #odd indices
    belong_to_surface2 = parent_child_lst[::2] #even indices
    
    if len(parent_child_lst)%2 != 0:
        string_fragment_surface2 = "Dom{0}".format(parent_child_lst[-1])
        
    string_diff_operation=[]
    
    if is_parent != []:
        index_even = list(range(0,len(parent_child_lst),2))
        for bol in range(int((len(parent_child_lst))/2)):
            string_diff_operation.append("geom.boolean_difference([Dom{0}],[Dom{1}])".format(parent_child_lst[index_even[bol]],parent_child_lst[index_even[bol]+1]))
            
        string_diff_operation_form = ("{0}".format(','.join(map(str,string_diff_operation))))
        
        if len(parent_child_lst)%2 != 0:
            inputList = ["geom.boolean_fragments([Dom0],["+string_diff_operation_form+","+string_domains_form+","+string_fragment_surface2+"])\n\n"]
        else:    
            inputList = ["geom.boolean_fragments([Dom0],["+string_diff_operation_form+","+string_domains_form+"])\n\n"]
    
    else: 
        inputList = ["geom.boolean_fragments([Dom0],["+string_domains_form+"])\n\n"] 
    
    string_star = "'*'"
    
    gmshFile.writelines(inputList)
    
    inputList = ['geom.add_raw_code("Mesh.SaveElementTagType=2;")\n\n','# Get the next available surface tag\n',
                 'geom.add_raw_code("LastSurf = news;")\n\n','# Rename surfaces resulting from Boolean operation(s) to ss\n',
                 'geom.add_raw_code("ss[] = Surface '+ string_star +';")\n\n']
    gmshFile.writelines(inputList)
    
    # List surfaces
    string_surf2 = ["s{0}".format(i) for i in sorted(lst_contours+belong_to_surface2)]
    string_surf_form2 = ("{0}".format(','.join(map(str,string_surf2))))
    
    string_length = repr(len(lst_contours+belong_to_surface2))
    
    filename_mesh = "{0}".format(filename)
    total_img_nmbr +=1
    
    inputList = ['geom.add_raw_code("Physical Surface(1) = {LastSurf-#ss[]+'+string_length+':LastSurf-1};")\n\n',
                 'geom.add_raw_code("Physical Surface(2) ={'+ string_surf_form2 +'};")\n\n',
                 'msh = pygmsh.generate_mesh(geom, geo_filename = "'+ filename_mesh +'.msh", prune_z_0=True)\n\n',
                 'tri_cells = np.array([None])\n','for cell in msh.cells:\n','   if cell.type == "triangle":\n',
                 '        # Collect the individual meshes\n','        if tri_cells.all() == None:\n',
                 '            tri_cells = cell.data\n','        else:\n',
                 '             tri_cells = np.concatenate((tri_cells, cell.data))\n\n', 'tri_data = None\n',
                 'for key in msh.cell_data_dict["gmsh:physical"].keys():\n', '    if key == "triangle":\n',
                 '       tri_data = msh.cell_data_dict["gmsh:physical"][key]\n', 
                 ' # Create triangular mesh with cell data attached\n', 
                 'tri_mesh = meshio.Mesh(points=msh.points, cells={"triangle": tri_cells},\n',
                 '                           cell_data={"SurfaceRegions":[tri_data]})\n\n',
                 'meshio.write("'+ filename_mesh +'.xdmf",tri_mesh)\n']
    
    gmshFile.writelines(inputList)
    
    gmshFile.close()