#create a spline basis function with pygismo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pygismo as gs  # Import pygismo

kv_u= gs.nurbs.gsKnotVector(np.array([0,0,0,0.25,0.50,0.75,1,1,1]), 2)
kv_v= gs.nurbs.gsKnotVector(np.array([0,0,0,0,0.25,0.50,0.75,1,1,1,1]), 3)

basis_u = gs.nurbs.gsBSplineBasis(kv_u)
basis_v = gs.nurbs.gsBSplineBasis(kv_v)

#opsione 1
tbasis= gs.nurbs.gsTensorBSplineBasis2(basis_u, basis_v)
print("the basis has size; ", tbasis.size())

#option 2
tbasis= gs.nurbs.gsTensorBSplineBasis2(kv_u, kv_v) 
print("the basis has size; ", tbasis.size())

N = M = 100
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, M)
XX, YY = np.meshgrid(x, y, indexing='xy')
pts = np.stack((XX.flatten(), YY.flatten()))

index = 30
z = tbasis.evalSingle(index, pts)
ZZ = z.reshape((N, M))

fig = plt.figure()
ax = fig.add_subplot( projection='3d')
ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm)
plt.show()




