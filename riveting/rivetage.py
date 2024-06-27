from fenics import *
from mshr import *


L = 15e-2; Ll = 5e-2; W = 0.4e-3; radius = 3e-3
head_radius = 4e-3; head_W = 2e-3; rivet_T = 2e-3
head_mandrel_radius = 1.5e-3; height_pop_rivet = W + 2*head_W
# Young's modulus and Poisson's ratio
E = 70e9
nu = 0.33

# Lame's constants
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu = E/2/(1+nu)

rho = 0 # à retirer
T = -20*9.81
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

tol = 1e-3

cyl = Cylinder(Point(L/2, Ll/2, 0), Point(L/2, Ll/2, height_pop_rivet), radius, radius )
head = Cylinder(Point(L/2, Ll/2, 0), Point(L/2, Ll/2, -head_W), head_radius, head_radius)

mandrel_cyl = Cylinder(Point(L/2, Ll/2, 0), Point(L/2, Ll/2, height_pop_rivet), radius-rivet_T, radius-rivet_T)
mandrel_head = Cylinder(Point(L/2, Ll/2, height_pop_rivet), Point(L/2, Ll/2, height_pop_rivet + head_W), head_mandrel_radius, head_mandrel_radius)

mandrel = mandrel_cyl + mandrel_head

domain = cyl + head - mandrel_cyl

polyhedral_domain = CSGCGALDomain3D(domain)

#print("Degenerate facets after boolean operation: {0}".format(polyhedral_domain.num_degenerate_facets(tol)))
#polyhedral_domain.remove_degenerate_facets(tol)

mesh = generate_mesh(domain, 32)
V = VectorFunctionSpace(mesh, 'P', 1)



def clamped_boundary_rivet_head(x, on_boundary):
    return on_boundary and x[2] < tol

def boundary_mandrel_cyl(x, on_boundary):
    return on_boundary and x[0]**2 + x[1]**2 - radius-rivet_T < tol


clamp_rivet_head = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary_rivet_head)
mandrel_cyl = DirichletBC(V, Expression(('0', '0', 'x[2]'), degree = 1), boundary_mandrel_cyl)
#tract = DirichletBC(V, Constant((0, T, 0)), traction_boundary)

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lambda_*div(u)*Identity(d) + 2*mu*epsilon(u)

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0, -rho*g))
#Tra = Constant((T, 0, 0))
Tra = Expression(('0', '0','x[2] >= Wc ? Tc : 0'), degree = 0, Wc = W, tolc = tol, Tc = T)
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(Tra, v)*ds

u = Function(V)
bcs = [clamp_rivet_head, mandrel_cyl]
solve(a == L, u, bcs)

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
