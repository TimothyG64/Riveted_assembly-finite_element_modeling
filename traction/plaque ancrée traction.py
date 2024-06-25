from fenics import *
from mshr import *


L = 15e-2; Ll = 5e-2; W = 0.4e-3; radius = 3e-3

# Young's modulus and Poisson's ratio
E = 70e9
nu = 0.33

# Lame's constants
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu = E/2/(1+nu)

rho = 0 # à retirer
T = 10*9.81
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

tol = 1e-5

plate = Box(Point(0,0,0), Point(L, Ll, W))
hole = Cylinder(Point(L/2, Ll/2, 0), Point(L/2, Ll/2, W), radius, radius )
domain = plate - hole

polyhedral_domain = CSGCGALDomain3D(domain)

#print("Degenerate facets after boolean operation: {0}".format(polyhedral_domain.num_degenerate_facets(tol)))
#polyhedral_domain.remove_degenerate_facets(tol)

mesh = generate_mesh(domain, 64)
V = VectorFunctionSpace(mesh, 'P', 1)



def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

def traction_boundary(x, on_boundary):
    return on_boundary and x[0] > L - tol

clamp = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)
#tract = DirichletBC(V, Constant((0, T, 0)), traction_boundary)

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lambda_*div(u)*Identity(d) + 2*mu*epsilon(u)

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0, -rho*g))
Tra = Constant((T, 0, 0))
Tra = Expression(('x[0] >= Lc-3e-2 ? Tc : 0', '0','0'), degree = 0, Lc = L, tolc = tol, Tc = T)
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(Tra, v)*ds

u = Function(V)
solve(a == L, u, clamp)

#plot(u, title='Displacement', mode='displacement')
File_Displacement = File('results/Displacement.pvd')
File_Displacement << u

s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
#plot(von_Mises, title='Stress intensity')
File_Stress = File('results/Stress.pvd')
File_Stress << von_Mises

u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
#plot(u_magnitude, 'Displacement magnitude')
File_DisplacementM = File('results/DisplacementMagnitude.pvd')
File_DisplacementM << u_magnitude
