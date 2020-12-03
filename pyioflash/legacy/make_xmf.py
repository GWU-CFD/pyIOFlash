import h5py
from xml.etree import ElementTree
from typing import Tuple, List, Set, Dict

def main() -> None:
    filenames = {'plot' : 'Poisson_hdf5_plt_cnt_0000', 
                 'grid' : 'Poisson_hdf5_grd_0000',
                 'xmf'  : 'Poisson'}
    write_xmf_file(create_xmf_file(filenames), filenames['xmf'])

def _first_true(iterable, predictor):
    return next(filter(predictor, iterable))

def create_xmf_file(filenames: Dict[str, str]) -> ElementTree.Element:
    blocks, sizes, fields = get_simulation_info(filenames['plot'])

    root = ElementTree.Element(*get_root_element())
    domain = ElementTree.SubElement(root, *get_domain_element())

    for block in range(blocks):
        grid = ElementTree.SubElement(domain, *get_grid_element(block))
        topology = ElementTree.SubElement(grid, *get_topology_element(sizes))

        geometry = ElementTree.SubElement(grid, *get_geometry_element())
        for axis in ('x', 'y', 'z'):
            hyperslab = ElementTree.SubElement(geometry, *get_geometry_hyperslab_header(sizes, axis))
            tag, attribute, text = get_geometry_hyperslab_slab(sizes, axis, block)
            ElementTree.SubElement(hyperslab, tag, attribute).text = text
            tag, attribute, text = get_geometry_hyperslab_data(sizes, axis, filenames['grid'])
            ElementTree.SubElement(hyperslab, tag, attribute).text = text

        for field in fields:
            Attribute = ElementTree.SubElement(grid, *get_attribute_element(field))
            hyperslab = ElementTree.SubElement(Attribute, *get_attribute_hyperslab_header(sizes))

            tag, attribute, text = get_attribute_hyperslab_slab(sizes, block)
            ElementTree.SubElement(hyperslab, tag, attribute).text = text

            tag, attribute, text = get_attribute_hyperslab_data(sizes, blocks, field, filenames['plot'])
            ElementTree.SubElement(hyperslab, tag, attribute).text = text

    return root


def write_xmf_file(root: ElementTree.Element, filename: str) -> None:
    with open(filename + '.xmf', 'wb') as file:
        file.write('<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'.encode('utf8'))
        ElementTree.ElementTree(root).write(file, short_empty_elements=False)

def get_simulation_info(filename: str) -> Tuple[int, Dict[str, int], Set[str]]:
    with h5py.File(filename, 'r') as file:
        int_scalars = list(file['integer scalars'])
        unknown_names = list(file['unknown names'][:, 0])
    blk_num = _first_true(int_scalars, lambda l: 'globalnumblocks' in str(l[0]))[1]
    blk_sizes = {i: _first_true(int_scalars, lambda l: 'n' + i + 'b' in str(l[0]))[1] for i in ('x', 'y', 'z')}
    fields = {k.decode('utf-8') for k in unknown_names}
    return blk_num, blk_sizes, fields

def get_comment_element() -> str:
    return 'DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []'

def get_root_element() -> Tuple[str, Dict[str, str]]:
    return ('Xdmf', {'xmlns:xi': 'http://www.w3.org/2003/XInclude', 'version': '2.2'})

def get_domain_element() -> Tuple[str, Dict[str, str]]:
    return ('Domain', {})

def get_grid_element(block: int) -> Tuple[str, Dict[str, str]]:
    return ('Grid', {'Name': str(block), 'GridType': 'Uniform'})

def get_topology_element(sizes: Dict[str, int]) -> Tuple[str, Dict[str, str]]:
    sizes = [sizes[axis] for axis in ('z', 'y', 'x')]
    dimensions = ' '.join([str(size + 1) for size in sizes])
    return ('Topology', {'Type': '3DRectMesh', 'NumberOfElements': dimensions})

def get_geometry_element() -> Tuple[str, Dict[str, str]]:
    return ('Geometry', {'Type': 'VXVYVZ'})

def get_geometry_hyperslab_header(sizes: Dict[str, int], axis: str) -> Tuple[str, Dict[str, str]]:
    dimensions = str(sizes[axis] + 1)
    return ('DataItem', {'ItemType': 'HyperSlab', 'Dimensions': dimensions, 'Type': 'HyperSlab'})

def get_geometry_hyperslab_slab(sizes: Dict[str, int], axis: str, block: int) -> Tuple[str, Dict[str, str], str]:
    size = sizes[axis] + 1
    dimensions = ' '.join(map(str, [block, 0, 1, 1, 1, size]))
    return ('DataItem', {'Dimensions': '3 2', 'NumberType': 'Int', 'Format': 'XML'}, dimensions)

def get_geometry_hyperslab_data(sizes: Dict[str, int], axis: str, filename: str) -> Tuple[str, Dict[str, str], str]:
    size = sizes[axis] + 1
    dimensions = ' '.join(map(str, [2, size]))
    filename = filename + ':/' + {'x': 'xxxf', 'y': 'yyyf', 'z': 'zzzf'}[axis]
    return ('DataItem', {'Format': 'HDF', 'Dimensions': dimensions, 'Name': axis,
                         'NumberType': 'Float', 'Precision': '4'}, filename)

def get_attribute_element(field: str) -> Tuple[str, Dict[str, str]]:
    return ('Attribute', {'Name': field, 'AttributeType': 'Scalar', 'Center': 'Cell'})

def get_attribute_hyperslab_header(sizes: Dict[str, int]) -> Tuple[str, Dict[str, str]]:
    sizes = [sizes[axis] for axis in ('z', 'y', 'x')]
    dimensions = ' '.join([str(size) for size in sizes])
    return ('DataItem', {'ItemType': 'HyperSlab', 'Dimensions': dimensions, 'Type': 'HyperSlab'})

def get_attribute_hyperslab_slab(sizes: Dict[str, int], block: int) -> Tuple[str, Dict[str, str], str]:
    sizes = [sizes[axis] for axis in ('z', 'y', 'x')]
    dimensions = ' '.join(map(str, [block, 0, 0, 0, 1, 1, 1, 1, 1] + sizes))
    return ('DataItem', {'Dimensions': '3 4', 'NumberType': 'Int', 'Format': 'XML'}, dimensions)

def get_attribute_hyperslab_data(sizes: Dict[str, int], blocks, field: str, filename: str) -> Tuple[str, Dict[str, str], str]:
    sizes = [sizes[axis] for axis in ('z', 'y', 'x')]
    dimensions = ' '.join(map(str, [blocks, ] + sizes))
    filename = filename + ':/' + field
    return ('DataItem', {'Format': 'HDF', 'Dimensions': dimensions, 'Name': field,
                         'NumberType': 'Float', 'Precision': '4'}, filename)

if __name__ == "__main__":
    main()

