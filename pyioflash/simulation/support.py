"""

This module defines the support methods and classes necessary for the pyio package.

Todo:
    None
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
            # xyz-directions ( {left || right} && {front || back} && up) 
            if "right" in faces and "front" in faces and "up" in faces:
                data[block, -g:, -g:, -g:] = data[struct[faces["right"]]["front"], g:2*g, g:2*g, g:2*g]
            if "left" in faces and "front" in faces and "up" in faces:
                data[block, -g:, -g:, :g] = data[struct[faces["left"]]["front"], g:2*g, g:2*g, -2*g:-g]
            if "right" in faces and "back" in faces and "up" in faces:
                data[block, -g:, :g, -g:] = data[struct[faces["right"]]["back"], g:2*g, -2*g:-g, g:2*g]
            if "left" in faces and "back" in faces and "up" in faces:
                data[block, -g:, :g, :g] = data[struct[faces["left"]]["back"], g:2*g, -2*g:-g, -2*g:-g]
                
            # xyz-directions ( {left || right} && {front || back} && down) 
            if "right" in faces and "front" in faces and "down" in faces:
                data[block, :g, -g:, -g:] = data[struct[faces["right"]]["front"], -2*g:-g, g:2*g, g:2*g]
            if "left" in faces and "front" in faces and "down" in faces:
                data[block, :g, -g:, :g] = data[struct[faces["left"]]["front"], -2*g:-g, g:2*g, -2*g:-g]
            if "right" in faces and "back" in faces and "down" in faces:
                data[block, :g, :g, -g:] = data[struct[faces["right"]]["back"], -2*g:-g, -2*g:-g, g:2*g]
            if "left" in faces and "back" in faces and "down" in faces:
                data[block, :g, :g, :g] = data[struct[faces["left"]]["back"], -2*g:-g, -2*g:-g, -2*g:-g]

def _bound_cells_from_data(data, geometry, field):
    """ Provides a method for filling boundary cells from data if hdf5 file does not contain such"""

    # apply boundary conditions to velocity
    if field in {'fcx2', 'fcy2', 'fcz2'}:
        _bound_cells_from_data_velc(data, geometry, field)

    # apply boundary conditions to temperature
    elif field == "temp":
        _bound_cells_from_data_temp(data, geometry)

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
    
    Implements method for velocity data

    """
    g = int(geometry.blk_guards / 2)
    struct = geometry.blk_neighbors

    for block, faces in enumerate(struct):

        # x-direction (left, right)
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
                print("neumann face!")
                if field == "fcx2":
                    print(block)
                    data[block, g:-g, g:-g, -g:] = data[block, g:-g, g:-g, -g-2:-2*g-2:-1]
                else:
                    data[block, g:-g, g:-g, -g:] = data[block, g:-g, g:-g, -g-1:-2*g-1:-1]
            else:
                pass
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

        # y-direction (front, back)
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

        if geometry.grd_dim == 3:
            # z-direction (up, down)
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

def _bound_cells_from_data_temp(data, geometry):
    """ Provides a method for filling boundary guards cells 
    from data if hdf5 file does not contain such
    
    Implements method for temperature data

    """
    g = int(geometry.blk_guards / 2)
    x = geometry._grd_mesh_x
    struct = geometry.blk_neighbors

    for block, faces in enumerate(struct):

        # x-direction (left, right)
        if "right" not in faces:
            if geometry.grd_bndcnds["temp"]["right"] in {"dirichlet_ht", "DIRICHLET_HT"}:
                data[block, g:-g, g:-g, -g:] = 2 * geometry.grd_bndvals["temp"]["right"] - \
                    data[block, g:-g, g:-g, -g-1:-2*g+1:-1]
            elif geometry.grd_bndcnds["temp"]["right"] in {"neumann_ht", "neumann_ht"}:
                data[block, g:-g, g:-g, -g:] = (x[block, g:-g, g:-g, -g:] - x[block, g:-g, g:-g, -g-1:-2*g+1:-1]) * \
                    geometry.grd_bndvals["temp"]["right"] + data[block, g:-g, g:-g, -g-1:-2*g+1:-1]
            else:
                pass

