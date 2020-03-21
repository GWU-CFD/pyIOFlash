"""

This module defines the support methods and classes necessary for the pyio package.

Todo:
    * Provide more correct and/or higher performance data filling
    * Eliminate branches in favor of casting for support methods

"""

import numpy

def _guard_cells_from_data(data, geometry):
    """ Provides a method for filling guard cells from data if hdf5 file does not contain such"""
    g = int(geometry.blk_guards / 2)
    struct = geometry.blk_neighbors

    for block, faces in enumerate(struct):

        # x-direction (left, right)
        if "right" in faces:
            data[block, g:-g, g:-g, -g:] = data[faces["right"], g:-g, g:-g, g:2*g]
        if "left" in faces:
            data[block, g:-g, g:-g, :g] = data[faces["left"], g:-g, g:-g, -2*g:-g]

        # y-direction (for, aft)
        if "front" in faces:
            data[block, g:-g, -g:, g:-g] = data[faces["front"], g:-g, g:2*g, g:-g]
        if "back" in faces:
            data[block, g:-g, :g, g:-g] = data[faces["back"], g:-g, -2*g:-g, g:-g]

        # z-direction (up, down)
        if "up" in faces:
            data[block, -g:, g:-g, g:-g] = data[faces["up"], g:2*g, g:-g, g:-g]
        if "down" in faces:
            data[block, :g, g:-g, g:-g] = data[faces["down"], -2*g:-g, g:-g, g:-g]

        # xy-directions ( {left || right} && {front || back} ) 
        if "right" in faces and "front" in faces:
            data[block, g:-g, -g:, -g:] = data[struct[faces["right"]]["front"], g:-g, g:2*g, g:2*g]
        if "left" in faces and "front" in faces:
            data[block, g:-g, -g:, :g] = data[struct[faces["left"]]["front"], g:-g, g:2*g, -2*g:-g]
        if "right" in faces and "back" in faces:
            data[block, g:-g, :g, -g:] = data[struct[faces["right"]]["back"], g:-g, -2*g:-g, g:2*g]
        if "left" in faces and "back" in faces:
            data[block, g:-g, :g, :g] = data[struct[faces["left"]]["back"], g:-g, -2*g:-g, -2*g:-g]

        if geometry.grd_dim == 3:
            
            # xz-directions ( {left || right} && {up || down} )
            if "right" in faces and "up" in faces:
                data[block, -g:, g:-g, -g:] = data[struct[faces["right"]]["up"], g:2*g, g:-g, g:2*g]
            if "left" in faces and "up" in faces:
                data[block, -g:, g:-g, :g] = data[struct[faces["left"]]["up"], g:2*g, g:-g, -2*g:-g]
            if "right" in faces and "down" in faces:
                data[block, :g, g:-g, -g:] = data[struct[faces["right"]]["down"], -2*g:-g, g:-g, g:2*g]
            if "left" in faces and "down" in faces:
                data[block, :g, g:-g, :g] = data[struct[faces["left"]]["down"], -2*g:-g, g:-g, -2*g:-g]

            # yz-directions ( {front || back} && {up || down} )
            if "front" in faces and "up" in faces:
                data[block, -g:, -g:, g:-g] = data[struct[faces["front"]]["up"], g:2*g, g:2*g, g:-g]
            if "back" in faces and "up" in faces:
                data[block, -g:, :g, g:-g] = data[struct[faces["back"]]["up"], g:2*g, -2*g:-g, g:-g]
            if "front" in faces and "down" in faces:
                data[block, :g, -g:, g:-g] = data[struct[faces["front"]]["down"], -2*g:-g, g:2*g, g:-g]
            if "back" in faces and "down" in faces:
                data[block, :g, :g, g:-g] = data[struct[faces["back"]]["down"], -2*g:-g, -2*g:-g, g:-g]

            # xyz-directions ( {left || right} && {front || back} && up) 
            if "right" in faces and "front" in faces and "up" in faces:
                data[block, -g:, -g:, -g:] = data[struct[struct[faces["right"]]["front"]]["up"], g:2*g, g:2*g, g:2*g]
            if "left" in faces and "front" in faces and "up" in faces:
                data[block, -g:, -g:, :g] = data[struct[struct[faces["left"]]["front"]]["up"], g:2*g, g:2*g, -2*g:-g]
            if "right" in faces and "back" in faces and "up" in faces:
                data[block, -g:, :g, -g:] = data[struct[struct[faces["right"]]["back"]]["up"], g:2*g, -2*g:-g, g:2*g]
            if "left" in faces and "back" in faces and "up" in faces:
                data[block, -g:, :g, :g] = data[struct[struct[faces["left"]]["back"]]["up"], g:2*g, -2*g:-g, -2*g:-g]
                
            # xyz-directions ( {left || right} && {front || back} && down) 
            if "right" in faces and "front" in faces and "down" in faces:
                data[block, :g, -g:, -g:] = data[struct[struct[faces["right"]]["front"]]["down"], -2*g:-g, g:2*g, g:2*g]
            if "left" in faces and "front" in faces and "down" in faces:
                data[block, :g, -g:, :g] = data[struct[struct[faces["left"]]["front"]]["down"], -2*g:-g, g:2*g, -2*g:-g]
            if "right" in faces and "back" in faces and "down" in faces:
                data[block, :g, :g, -g:] = data[struct[struct[faces["right"]]["back"]]["down"], -2*g:-g, -2*g:-g, g:2*g]
            if "left" in faces and "back" in faces and "down" in faces:
                data[block, :g, :g, :g] = data[struct[struct[faces["left"]]["back"]]["down"], -2*g:-g, -2*g:-g, -2*g:-g]

def _bound_cells_from_data(data, geometry, field):
    """ Provides a method for filling boundary cells from data if hdf5 file does not contain such"""

    # apply boundary conditions to momentum-like face centered fields
    if field in {'fcx2', 'fcy2', 'fcz2'}:
        _bound_cells_from_data_velc(data, geometry, field)

    # apply boundary conditions to inhomogeious cell centered fields
    elif field in {"temp"}:
        _bound_cells_from_data_temp(data, geometry, field)

    # apply boundary conditions to homogenious cell centered fields
    elif field in {"pres", "dust"}:
        _bound_cells_from_data_pres(data, geometry, field)

    # fill boundary guard cells for grid meshes
    elif field in {"xxxl", "xxxc", "xxxr", "yyyl", "yyyc", "yyyr", "zzzl", "zzzc", "zzzr"}:
        _bound_cells_from_data_grid(data, geometry, field)

    # fill boundary guard cells for grid metric meshes
    elif field in {"ddxl", "ddxc", "ddxr", "ddyl", "ddyc", "ddyr", "ddzl", "ddzc", "ddzr"}:
        _bound_cells_from_data_metric(data, geometry, field)

    else:
        pass

def _bound_cells_from_data_grid(data, geometry, field):
    """ Provides a method for filling boundary guards cells 
    from data if hdf5 file does not contain such
    
    Implements method for grid mesh data

    """
    g = int(geometry.blk_guards / 2)
    l = numpy.linspace(1, g, g)
    struct = geometry.blk_neighbors

    for block, faces in enumerate(struct):

        # x-direction (left, right)
        if "right" not in faces:
            for i in range(g):
                data[block, :, :, -i-1] = data[block, :, :, -g-1] + (data[block, :, :, -g-1] - data[block, :, :, -g-2]) * (g - i)
        if "left" not in faces:
            for i in range(g):
                data[block, :, :, i] = data[block, :, :, g] - (data[block, :, :, g+1] - data[block, :, :, g]) * (g - i)

        # y-direction (front, back)
        if "front" not in faces:
            for i in range(g):
                data[block, :, -i-1, :] = data[block, :, -g-1, :] + (data[block, :, -g-1, :] - data[block, :, -g-2, :]) * (g - i)
        if "back" not in faces:
            for i in range(g):
                data[block, :, i, :] = data[block, :, g, :] - (data[block, :, g+1, :] - data[block, :, g, :]) * (g - i)

        if geometry.grd_dim == 3:
            # z-direction (up, down)
            if "up" not in faces:
                for i in range(g):
                    data[block, -i-1, :, :] = data[block, -g-1, :, :] + (data[block, -g-1, :, :] - data[block, -g-2, :, :]) * (g - i)
            if "down" not in faces:
                for i in range(g):
                    data[block, i, :, :] = data[block, g, :, :] - (data[block, g+1, :, :] - data[block, g, :, :]) * (g - i)

def _bound_cells_from_data_metric(data, geometry, field):
    """ Provides a method for filling boundary guards cells 
    from data if hdf5 file does not contain such
    
    Implements method for grid metric mesh data

    """
    g = int(geometry.blk_guards / 2)
    struct = geometry.blk_neighbors

    for block, faces in enumerate(struct):

        # x-direction (left, right)
        if "right" not in faces:
            for i in range(g):
                data[block, :, :, -i-1] = data[block, :, :, -g-1]
        if "left" not in faces:
            for i in range(g):
                data[block, :, :, i] = data[block, :, :, g]

        # y-direction (front, back)
        if "front" not in faces:
            for i in range(g):
                data[block, :, -i-1, :] = data[block, :, -g-1, :]
        if "back" not in faces:
            for i in range(g):
                for i in range(g):
                    data[block, :, i, :] = data[block, :, g, :]

        if geometry.grd_dim == 3:
            # z-direction (up, down)
            if "up" not in faces:
                for i in range(g):
                    data[block, -i-1, :, :] = data[block, -g-1, :, :]
            if "down" not in faces:
                for i in range(g):
                    data[block, i, :, :] = data[block, g, :, :]

def _bound_cells_from_data_velc(data, geometry, field):
    """ Provides a method for filling boundary guards cells 
    from data if hdf5 file does not contain such
    
    Implements method for velocity and other momentum-like
    face centered fields

    """
    g = int(geometry.blk_guards / 2)
    struct = geometry.blk_neighbors

    for block, faces in enumerate(struct):

        # x-direction (right)
        if "right" not in faces:
            if geometry.grd_bndcnds["velc"]["right"] in {"noslip_ins", "NOSLIP_INS"}:
                data[block, g:-g, g:-g, -g:] = 0.0
            elif geometry.grd_bndcnds["velc"]["right"] in {"slip_ins", "SLIP_INS"}:
                if field == "fcx2":
                    data[block, g:-g, g:-g, -g:] = 0.0
                else:
                    data[block, g:-g, g:-g, -g:] = data[block, g:-g, g:-g, -g-1:-2*g-1:-1]
            elif geometry.grd_bndcnds["velc"]["right"] in {"neumann", "NEUMANN",
                                                            "neumann_ins", "NEUMANN_INS"}:
                if field == "fcx2":
                    data[block, g:-g, g:-g, -g:] = data[block, g:-g, g:-g, -g-2:-2*g-2:-1]
                else:
                    data[block, g:-g, g:-g, -g:] = data[block, g:-g, g:-g, -g-1:-2*g-1:-1]
            else:
                pass

        # x-direction (left)
        if "left" not in faces:
            if geometry.grd_bndcnds["velc"]["left"] in {"noslip_ins", "NOSLIP_INS"}:
                data[block, g:-g, g:-g, :g] = 0.0
            elif geometry.grd_bndcnds["velc"]["left"] in {"slip_ins", "SLIP_INS"}:
                if field == "fcx2":
                    data[block, g:-g, g:-g, :g] = 0.0
                else:
                    data[block, g:-g, g:-g, :g] = data[block, g:-g, g:-g, 2*g-1:g-1:-1]
            elif geometry.grd_bndcnds["velc"]["left"] in {"neumann", "NEUMANN",
                                                            "neumann_ins", "NEUMANN_INS"}:
                if field == "fcx2":
                    data[block, g:-g, g:-g, :g-1] = data[block, g:-g, g:-g, 2*g-2:g-1:-1]
                else:
                    data[block, g:-g, g:-g, :g] = data[block, g:-g, g:-g, 2*g-1:g-1:-1]
            else:
                pass

        # y-direction (front)
        if "front" not in faces:
            if geometry.grd_bndcnds["velc"]["front"] in {"noslip_ins", "NOSLIP_INS"}:
                data[block, g:-g, -g:, g:-g] = 0.0
            elif geometry.grd_bndcnds["velc"]["front"] in {"slip_ins", "SLIP_INS"}:
                if field == "fcy2":
                    data[block, g:-g, -g:, g:-g] = 0.0
                else:
                    data[block, g:-g, -g:, g:-g] = data[block, g:-g, -g-1:-2*g-1:-1, g:-g]
            elif geometry.grd_bndcnds["velc"]["front"] in {"neumann", "NEUMANN",
                                                            "neumann_ins", "NEUMANN_INS"}:
                if field == "fcy2":
                    data[block, g:-g, -g:, g:-g] = data[block, g:-g, -g-2:-2*g-2:-1, g:-g]
                else:
                    data[block, g:-g, -g:, g:-g] = data[block, g:-g, -g-1:-2*g-1:-1, g:-g]
            else:
                pass

        # y-direction (back)
        if "back" not in faces:
            if geometry.grd_bndcnds["velc"]["back"] in {"noslip_ins", "NOSLIP_INS"}:
                data[block, g:-g, :g, g:-g] = 0.0
            elif geometry.grd_bndcnds["velc"]["back"] in {"slip_ins", "SLIP_INS"}:
                if field == "fcy2":
                    data[block, g:-g, :g, g:-g] = 0.0
                else:
                    data[block, g:-g, :g, g:-g] = data[block, g:-g, 2*g-1:g-1:-1, g:-g]
            elif geometry.grd_bndcnds["velc"]["back"] in {"neumann", "NEUMANN",
                                                            "neumann_ins", "NEUMANN_INS"}:
                if field == "fcy2":
                    data[block, g:-g, :g-1, g:-g] = data[block, g:-g, 2*g-2:g-1:-1, g:-g]
                else:
                    data[block, g:-g, :g, g:-g] = data[block, g:-g, 2*g-1:g-1:-1, g:-g]
            else:
                pass

        # only apply conditions for 3D if necessary
        if geometry.grd_dim == 3:

            # z-direction (up)
            if "up" not in faces:
                if geometry.grd_bndcnds["velc"]["up"] in {"noslip_ins", "NOSLIP_INS"}:
                    data[block, -g:, g:-g, g:-g] = 0.0
                elif geometry.grd_bndcnds["velc"]["up"] in {"slip_ins", "SLIP_INS"}:
                    if field == "fcz2":
                        data[block, -g:, g:-g, g:-g] = 0.0
                    else:
                        data[block, -g:, g:-g, g:-g] = data[block, -g-1:-2*g-1:-1, g:-g, g:-g]
                elif geometry.grd_bndcnds["velc"]["up"] in {"neumann", "NEUMANN",
                                                                "neumann_ins", "NEUMANN_INS"}:
                    if field == "fcz2":
                        data[block, -g:, g:-g, g:-g] = data[block, -g-2:-2*g-2:-1, g:-g, g:-g]
                    else:
                        data[block, -g:, g:-g, g:-g] = data[block, -g-1:-2*g-1:-1, g:-g, g:-g]
                else:
                    pass

            # z-direction (down)
            if "down" not in faces:
                if geometry.grd_bndcnds["velc"]["down"] in {"noslip_ins", "NOSLIP_INS"}:
                    data[block, :g, g:-g, g:-g] = 0.0
                elif geometry.grd_bndcnds["velc"]["down"] in {"slip_ins", "SLIP_INS"}:
                    if field == "fcz2":
                        data[block, :g, g:-g, g:-g] = 0.0
                    else:
                        data[block, :g, g:-g, g:-g] = data[block, 2*g-1:g-1:-1, g:-g, g:-g]
                elif geometry.grd_bndcnds["velc"]["down"] in {"neumann", "NEUMANN",
                                                                "neumann_ins", "NEUMANN_INS"}:
                    if field == "fcz2":
                        data[block, :g-1, g:-g, g:-g] = data[block, 2*g-2:g-1:-1, g:-g, g:-g]
                    else:
                        data[block, :g, g:-g, g:-g] = data[block, 2*g-1:g-1:-1, g:-g, g:-g]
                else:
                    pass

def _bound_cells_from_data_temp(data, geometry, field):
    """ Provides a method for filling boundary guards cells 
    from data if hdf5 file does not contain such
    
    Implements method for cell centered fields 
    with inhomogenious boundary values

    Requires boundary condition data to be available 
    in the GeometryData object for the specified field

    """
    g = int(geometry.blk_guards / 2)
    x = geometry._grd_mesh_x[1]
    y = geometry._grd_mesh_y[1]
    z = geometry._grd_mesh_z[1]
    struct = geometry.blk_neighbors



    # fill all boundary faces in first cycle
    for block, faces in enumerate(struct):

        # x-direction (right)
        if "right" not in faces:
            if geometry.grd_bndcnds[field]["right"] in {"dirichlet_ht", "DIRICHLET_HT"}:
                data[block,:,:,-g:] = 2 * geometry.grd_bndvals[field]["right"] - data[block,:,:,-g-1:-2*g-1:-1]
            elif geometry.grd_bndcnds[field]["right"] in {"neumann_ht", "neumann_ht"}:
                data[block,:,:,-g:] = (x[block,:,:,-g:] - x[block,:,:,-g-1:-2*g-1:-1]) * geometry.grd_bndvals[field]["right"] + data[block,:,:,-g-1:-2*g-1:-1]
            else:
                pass

        # x-direction (left)
        if "left" not in faces:
            if geometry.grd_bndcnds[field]["left"] in {"dirichlet_ht", "DIRICHLET_HT"}:
                data[block,:,:,:g] = 2 * geometry.grd_bndvals[field]["left"] - data[block, :, :,2*g-1:g-1:-1]
            elif geometry.grd_bndcnds[field]["left"] in {"neumann_ht", "neumann_ht"}:
                data[block,:,:,:g] = (x[block,:,:,2*g-1:g-1:-1] - x[block,:,:,:g]) * geometry.grd_bndvals[field]["left"] + data[block,:,:,2*g-1:g-1:-1]
            else:
                pass

        # y-direction (front)
        if "front" not in faces:
            if geometry.grd_bndcnds[field]["front"] in {"dirichlet_ht", "DIRICHLET_HT"}:
                data[block,:,-g:,:] = 2 * geometry.grd_bndvals[field]["front"] - data[block,:,-g-1:-2*g-1:-1,:]
            elif geometry.grd_bndcnds[field]["front"] in {"neumann_ht", "neumann_ht"}:
                data[block,:,-g:,:] = (y[block,:,-g:,:] - y[block,:,-g-1:-2*g-1:-1,:]) * geometry.grd_bndvals[field]["front"] + data[block,:,-g-1:-2*g-1:-1,:]
            else:
                pass

        # y-direction (back)
        if "back" not in faces:
            if geometry.grd_bndcnds[field]["back"] in {"dirichlet_ht", "DIRICHLET_HT"}:
                data[block,:,:g,:] = 2 * geometry.grd_bndvals[field]["back"] - data[block,:,2*g-1:g-1:-1,:]
            elif geometry.grd_bndcnds[field]["back"] in {"neumann_ht", "neumann_ht"}:
                data[block,:,:g,:] = (y[block,:,2*g-1:g-1:-1,:] - y[block,:,:g,:]) * geometry.grd_bndvals[field]["back"] + data[block,:,2*g-1:g-1:-1,:]
            else:
                pass

        # only apply conditions for 3D if necessary
        if geometry.grd_dim == 3:

            # z-direction (up)
            if "up" not in faces:
                if geometry.grd_bndcnds[field]["up"] in {"dirichlet_ht", "DIRICHLET_HT"}:
                    data[block,-g:,g:-g,g:-g] = 2 * geometry.grd_bndvals[field]["up"] - data[block,-g-1:-2*g-1:-1,g:-g,g:-g]
                elif geometry.grd_bndcnds[field]["up"] in {"neumann_ht", "neumann_ht"}:
                    data[block,-g:,g:-g,g:-g] = (z[block,-g:,g:-g,g:-g] - z[block,-g-1:-2*g-1:-1,g:-g,g:-g]) * geometry.grd_bndvals[field]["up"] + data[block,-g-1:-2*g-1:-1,g:-g,g:-g]
                else:
                    pass

            # z-direction (down)
            if "down" not in faces:
                if geometry.grd_bndcnds[field]["down"] in {"dirichlet_ht", "DIRICHLET_HT"}:
                    data[block, g:-g, g:-g, :g] = 2 * geometry.grd_bndvals[field]["down"] - \
                        data[block, g:-g, g:-g, 2*g-1:g-1:-1]
                elif geometry.grd_bndcnds[field]["down"] in {"neumann_ht", "neumann_ht"}:
                    data[block, :g, g:-g, g:-g] = (z[block, 2*g-1:g-1:-1, g:-g, g:-g] - z[block, :g, g:-g, g:-g]) * \
                        geometry.grd_bndvals[field]["down"] + data[block, 2*g-1:g-1:-1, g:-g, g:-g]
                else:
                    pass

        # fill all z boundary faces (xy internally bounded) in second cycle
        for block, faces in enumerate(struct):

            # xz-directions ( {left || right} && {up || down} )
            if "right" in faces and "up" not in faces:
                data[block, -g:, g:-g, -g:] = data[faces["right"], -g:, g:-g, g:2*g]



    # fill all xz and yz planer boundary faces in second cycle
    for block, faces in enumerate(struct):



        # xy-directions ( {left || right} && {front || back} )
        # correct the domain corners as avg of the directional ghost cells
        if "right" not in faces and "front" not in faces:
            data[block, :, -g:, -g:] = data[block, :, -g:, -g-1] / 2
            data[block, :, -g:, -g:] += data[block, :, -g-1, -g:] / 2
        if "left" not in faces and "front" not in faces:
            data[block, :, -g:, :g] = data[block, :, -g:, g] / 2
            data[block, :, -g:, :g] += data[block, :, -g-1, :g] / 2
        if "right" not in faces and "back" not in faces:
            data[block, :, :g, -g:] = data[block, :, :g, -g-1] / 2
            data[block, :, :g, -g:] += data[block, :, g, -g:] / 2
        if "left" not in faces and "back" not in faces:
            data[block, :, :g, :g] = data[block, :, :g, g] / 2
            data[block, :, :g, :g] += data[block, :, g, :g] / 2


def _bound_cells_from_data_pres(data, geometry, field):
    """ Provides a method for filling boundary guards cells 
    from data if hdf5 file does not contain such
    
    Implements method for cell centered fields 
    with homogenious boundary values

    """
    g = int(geometry.blk_guards / 2)
    struct = geometry.blk_neighbors

    print(field + " in _pres support")
    field = "velc"
    for block, faces in enumerate(struct):

        # x-direction (right)
        if "right" not in faces:
            if geometry.grd_bndcnds[field]["right"] in {"dirichlet", "DIRICHLET"}:
                data[block, g:-g, g:-g, -g:] = 2 * 0.0 - data[block, g:-g, g:-g, -g-1:-2*g-1:-1]
            elif geometry.grd_bndcnds[field]["right"] in {"neumann",    "neumann",
                                                          "noslip_ins", "NOSLIP_INS",
                                                          "slip_ins",   "SLIP_INS",
                                                          "movlid_ins", "MOVLID_INS"}:
                data[block, g:-g, g:-g, -g:] = 0.0 + data[block, g:-g, g:-g, -g-1:-2*g-1:-1]
            else:
                pass

        # x-direction (left)
        if "left" not in faces:
            if geometry.grd_bndcnds[field]["right"] in {"dirichlet", "DIRICHLET"}:
                data[block, g:-g, g:-g, :g] = 2 * 0.0 - data[block, g:-g, g:-g, 2*g-1:g-1:-1]
            elif geometry.grd_bndcnds[field]["right"] in {"neumann",    "neumann",
                                                          "noslip_ins", "NOSLIP_INS",
                                                          "slip_ins",   "SLIP_INS",
                                                          "movlid_ins", "MOVLID_INS"}:
                data[block, g:-g, g:-g, :g] = 0.0 + data[block, g:-g, g:-g, 2*g-1:g-1:-1]
            else:
                pass

        # y-direction (front)
        if "front" not in faces:
            if geometry.grd_bndcnds[field]["front"] in {"dirichlet", "DIRICHLET"}:
                data[block, g:-g, -g:, g:-g] = 2 * 0.0 - data[block, g:-g, -g-1:-2*g-1:-1, g:-g]
            elif geometry.grd_bndcnds[field]["front"] in {"neumann",    "neumann",
                                                          "noslip_ins", "NOSLIP_INS",
                                                          "slip_ins",   "SLIP_INS",
                                                          "movlid_ins", "MOVLID_INS"}:
                data[block, g:-g, -g:, g:-g] = 0.0 + data[block, g:-g, -g-1:-2*g-1:-1, g:-g]
            else:
                pass

        # y-direction (back)
        if "back" not in faces:
            if geometry.grd_bndcnds[field]["back"] in {"dirichlet", "DIRICHLET"}:
                data[block, g:-g, :g, g:-g] = 2 * 0.0 - data[block, g:-g, 2*g-1:g-1:-1, g:-g]
            elif geometry.grd_bndcnds[field]["back"] in {"neumann",    "neumann",
                                                         "noslip_ins", "NOSLIP_INS",
                                                         "slip_ins",   "SLIP_INS",
                                                          "movlid_ins", "MOVLID_INS"}:
                data[block, g:-g, :g, g:-g] = 0.0 + data[block, g:-g, 2*g-1:g-1:-1, g:-g]
            else:
                pass

        # only apply conditions for 3D if necessary
        if geometry.grd_dim == 3:

            # z-direction (up)
            if "up" not in faces:
                if geometry.grd_bndcnds[field]["up"] in {"dirichlet", "DIRICHLET"}:
                    data[block, -g:, g:-g, g:-g] = 2 * 0.0 - data[block, -g-1:-2*g-1:-1, g:-g, g:-g]
                elif geometry.grd_bndcnds[field]["up"] in {"neumann",    "neumann",
                                                          "noslip_ins", "NOSLIP_INS",
                                                          "slip_ins",   "SLIP_INS",
                                                          "movlid_ins", "MOVLID_INS"}:
                    data[block, -g:, g:-g, g:-g] = 0.0 + data[block, -g-1:-2*g-1:-1, g:-g, g:-g]
                else:
                    pass

            # z-direction (down)
            if "down" not in faces:
                if geometry.grd_bndcnds[field]["down"] in {"dirichlet", "DIRICHLET"}:
                    data[block, g:-g, g:-g, :g] = 2 * 0.0 - data[block, g:-g, g:-g, 2*g-1:g-1:-1]
                elif geometry.grd_bndcnds[field]["down"] in {"neumann",    "neumann",
                                                             "noslip_ins", "NOSLIP_INS",
                                                             "slip_ins",   "SLIP_INS",
                                                          "movlid_ins", "MOVLID_INS"}:
                    data[block, :g, g:-g, g:-g] = 0.0 + data[block, 2*g-1:g-1:-1, g:-g, g:-g]
                else:
                    pass
