from pyioflash import SimulationData
from matplotlib import pyplot

data = SimulationData.from_list([0], path="../../../", basename="INS_Taylor_", header="hdf5_chk_", form="chk")

z = 1
geometry = data.geometry
xxxc = geometry._grd_mesh_x[1, :, z, :, :]
yyyc = geometry._grd_mesh_y[1, :, z, :, :]
xxxr = geometry._grd_mesh_x[2, :, z, :, :]
yyyr = geometry._grd_mesh_x[2, :, z, :, :]
ddxc = geometry._grd_mesh_ddx[1, :, z, :, :]
ddyc = geometry._grd_mesh_ddy[1, :, z, :, :]

fields = data.fields.get(0)

temp = fields._temp[:, z, :, :]
tmax = fields._temp_max
tmin = fields._temp_min

dust = fields._dust[:, z, :, :]
dmax = fields._dust_max
dmin = fields._dust_min

pres = fields._pres[:, z, :, :]
pmax = fields._pres_max
pmin = fields._pres_min

fcx2 = fields._fcx2[:, z, :, :]
fcy2 = fields._fcy2[:, z, :, :]

fig = pyplot.figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for b in [0, 1, 2, 3]:
    pyplot.scatter(xxxc[b], yyyc[b], c=temp[b], cmap='viridis', vmin=tmin, vmax=tmax)

pyplot.grid()
pyplot.colorbar()
#pyplot.xlim(geometry.grd_bndbox[0])
#pyplot.ylim(geometry.grd_bndbox[1])
pyplot.show()