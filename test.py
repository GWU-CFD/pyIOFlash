from pyioflash import SimulationData
from matplotlib import pyplot

data = SimulationData.from_list([0], path="../../../", basename="INS_Couette_", header="hdf5_chk_", form="chk")

x = data.geometry._grd_mesh_x[1, :, 1, :, :]
y = data.geometry._grd_mesh_y[1, :, 1, :, :]
ddx = data.geometry._grd_mesh_ddx[1, :, 1, :, :]
t = data.fields.tolist()[0]._temp[:, 1, :, :]

xu = data.geometry._grd_mesh_x[2, :, 1, :, :]
yv = data.geometry._grd_mesh_x[2, :, 1, :, :]
u = data.fields.tolist()[0]._fcx2[:, 1, :, :]
v = data.fields.tolist()[0]._fcy2[:, 1, :, :]

fig = pyplot.figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for b in range(4):
    pyplot.contourf(xu[b], y[b], u[b], 200, vmin=0, vmax=1.0, cmap="viridis")

pyplot.xlim((0, 3.14159*2))
pyplot.ylim((0, 3.14159*2))
pyplot.colorbar()
pyplot.show()