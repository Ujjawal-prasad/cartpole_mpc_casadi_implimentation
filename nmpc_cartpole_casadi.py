import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

#=======================
# PARAMETERS
#=======================

dt = 0.02          # control frequency
N = 20             # optimization horizon

mc = 1.0           # mass of the cart
mp = 0.1           # mass of the pole
l = 0.5            # pole lemgth
g = 9.81           # gravity

nx = 4             # number of states
nu = 1             # number of control input

#=======================
# SYMBOLIC STATES
#=======================

x = ca.SX.sym('x')
x_dot = ca.SX.sym('x_dot')
theta = ca.SX.sym('theta')
theta_dot = ca.SX.sym('theta_dot')

states = ca.vertcat(x,x_dot,theta,theta_dot)

u = ca.SX.sym('u')

#=======================
# Dynamics
#=======================

sin_t = ca.sin(theta)
cos_t = ca.cos(theta)

den = mc + mp * sin_t**2

x_ddot = ( u + mp * sin_t * (l * theta_dot**2 + g * cos_t )) /den
theta_ddot = ( -u * cos_t - mp * l * theta_dot**2 * cos_t * sin_t - ( mc + mp ) * g * sin_t) /( l * den)

rhs = ca.vertcat(x_dot,x_ddot,theta_dot,theta_ddot)

f = ca.Function('f',[states,u],[rhs])

#=======================
#OPTIMIZATION VARIABLES
#=======================

X = ca.SX.sym('X', nx, N+1)
U = ca.SX.sym('U', nu, N)

#parameters
#current state + target states
P = ca.SX.sym('P',2 * nx)

#=======================
# COST MATRICES
#=======================

Q = ca.diag([10,1,100,1])
R = ca.diag([0.1])

#=======================
# OBJECTIVE + CONSTRAINTS
#=======================

obj = 0
g_con = []

# initial condition constraint
g_con.append(X[:,0] - P[:nx])

for k in range(N):
    st = X[:,k]
    con = U[:,k]
    ref = P[nx:]

    err = st - ref

    obj+= ca.mtimes([err.T,Q,err])
    obj+= ca.mtimes([con.T,R,con])

    st_next = X[:,k+1]

    f_value = f(st, con)

    st_next_euler = st + dt * f_value

    g_con.append(st_next - st_next_euler)

#==========================
#OPTIMISATION VARIABLES
#==========================

OPT_variables = ca.vertcat(
    ca.reshape(X,-1,1),
    ca.reshape(U,-1,1)
)
g_con = ca.vertcat(*g_con)

#==========================
#NLP
#==========================

nlp_prob = {
    'f': obj,
    'x': OPT_variables,
    'g': g_con,
    'p': P
}

solver = ca.nlpsol(
    'solver',
    'ipopt',
    nlp_prob
)

#===========================
#BOUNDS
#===========================
lbg = np.zeros(g_con.shape[0])
ubg = np.zeros(g_con.shape[0])

lbx = []
ubx = []

#state bounds
for _ in range(N + 1):
    lbx += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
    ubx += [ ca.inf,  ca.inf,  ca.inf,  ca.inf]

#control bounds
for _ in range(N):
    lbx += [-10]
    ubx += [ 10]

#===========================
#SIMULATION
#===========================

x0 = np.array([0, 0, 3.14, 0])
x_ref = np.array([0, 0, 0, 0])

sim_time = 3000

history = []
control_history = []

for i in range(sim_time):
    args_p = np.concatenate((x0,x_ref))

    init_guess = np.zeros(OPT_variables.shape[0])

    sol = solver(
        x0 = init_guess,
        lbx = lbx,
        ubx = ubx,
        lbg = lbg,
        ubg = ubg,
        p = args_p  
    )

    sol_x = sol['x'].full()
    offset_u = nx * (N + 1)
    u0 = sol_x[offset_u]
    dx = f(x0, u0).full().flatten()
    x0 = x0 + dt * dx

    history.append(x0.copy())
    control_history.append(u0.copy())

history = np.array(history)
control_history = np.array(control_history)

#============================
#PLOT
#============================

plt.plot(history[:,0])
plt.title("Displacement")
plt.xlabel("Time step")
plt.ylabel("Theta")
plt.grid()
plt.show()

plt.plot(history[:,1])
plt.title("Velocity")
plt.xlabel("Time step")
plt.ylabel("Theta")
plt.grid()
plt.show()

plt.plot(history[:,2])
plt.title("Pole Angle")
plt.xlabel("Time step")
plt.ylabel("Theta")
plt.grid()
plt.show()

plt.plot(history[:,3])
plt.title("Angular velocity")
plt.xlabel("Time step")
plt.ylabel("Theta")
plt.grid()
plt.show()

plt.plot(control_history[:])
plt.title("Control")
plt.xlabel("Time step")
plt.ylabel("Theta")
plt.grid()
plt.show()