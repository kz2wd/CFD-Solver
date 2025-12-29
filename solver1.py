import numpy as np


size = (10, 10)


t = 0.1  # Time
u = np.zeros((*size, 2))  # Velocity
rho = 1  # Density
p = [0 for _ in range(1)]  # Pressure
v = [0 for _ in range(1)]  # Viscosity

dx = 1

# we have initial u and p and??? and we want them after X steps 

# u(x, y, t) -> (x, y)

# du/dt + u . D u = -1/p D p + v D² u
# du/dt : how u changes when time changes
# u . D u : D u is how u will change, so u . D u is the current state time how it will change, so it's the new state



delta
steps = 100

for s in range(steps):
    pass

