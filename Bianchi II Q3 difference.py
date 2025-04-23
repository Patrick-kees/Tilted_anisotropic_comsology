import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy import optimize
import time

# Constants
w = 1/3  #speed of sound sqaured
O = 0.15 #Hubble normalized density of the universe

# Start the timer for the event
start_time = time.time()

def time_limit_event(t, y):
    # Stop after 600 seconds (10 minutes)
    if time.time() - start_time > 600000:
        return 0
    return 1

time_limit_event.terminal = True

# Model functions
def Q_three(params):
    sc, no = params
    return -(no * sc) / np.sqrt(3)

def Omega_fun(params):
    sp, sm, sc, sa, no = params
    return 1 - (sp**2 + sm**2 + sc**2 +sa**2) - (no**2) / 12

def tilted_velocity(params):
    sp, sm, sc, sa, no = params
    params_q3 = [sc, no]

    Q = Q_three(params_q3)
    O = Omega_fun(params)
    
    A = (1 + w) * O
    B = ((1 + w)**2) * O**2
    C = 4 * w * (Q)**2

    tilted_velocity = (2 * Q) / (A + np.sqrt(np.abs(B - C)))
    return tilted_velocity

def q(params):
    sp, sm, sc, sa, no = params
    params_q3 = [sc, no]

    Q = Q_three(params_q3)
    v_tilt = tilted_velocity(params)
    O = Omega_fun(params)

    SigmaSquared = sp**2+sm**2+sc**2+sa**2

    q_val = 2 * SigmaSquared + 0.5 * (1 + 3*w) * O + 0.5 * (1 - 3*w) * Q * v_tilt
    return q_val

#Constraints
    #Density evolution equation difference
def Constraint_1(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sp, sm, sc, sa, no]

    q_val = q(params_q)
    O = Omega_fun(params_q)

    OE1 = 2*(q_val-2)*O+3*(1-w)*O+(1-3*w-sp-np.sqrt(3)*sm)*(no*sc*v)*(1/np.sqrt(3))

    OE2P1 = 2*(2-q_val)*(1-O)-(1/3)*(1-4*sp)*no**2-2*(6*sp-np.sqrt(3)*sm)*sc**2
    OE2P2 = ((1+w)/(w+w*v**2))*(O**2)*(v**2)*(np.sqrt(3)*sm-sp)
    OE2 = OE2P1 + OE2P2

    Difference = OE1 - OE2

    return Difference
    #Q_3 equation difference
def Constraint_2(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sp, sm, sc, sa, no]
    params_O = [sp, sm, sc, sa, no]
    params_q3 = [sc, no]

    q_val = q(params_q)
    O = Omega_fun(params_O)

    v_def = (1+w)/(1+w*v**2)*O*v

    Difference = np.abs(Q_three(params_q3) - v_def)

    return Difference

"""#SM function (rearranged Constraint_1)
def SM(params):

    sp, sc, sa, no, v = params
    params_q = [sp, sc, sa, no]
    q_val = q(params_q)

    q_val = q(sp, sc, sa, no)
    vf = (1+w)/(1+w*v**2)*(O*v)**2

    A = 3*(1-w)*O+(1-3*w-sp)*no*sc*v/np.sqrt(3)
    B = 2*(q_val-2)*(1-2*O)+(1-4*sp)*(no**2)/3+12*sp*sc**2
    C = vf*sp
    D =no*sc*v+np.sqrt(3)*(2*sc**2+vf)
    return (A+B+C)/D

#SP function (rearranged Constraint_1)
def SP(params):
    sm, sc, sa, no = params
    v = tilted_velocity(params)
    params_q = [sm, sc, sa, no]
    q_val = q(params_q)

    vf = (1+w)/(1+w*v**2)*(O*v)**2
    A = 2*(2-q_val)-(no**2)/3+np.sqrt(12)*sm*sc**2
    B = np.sqrt(3)*vf*sm*(v*O)**2
    C = -3*(1-w)*O-(1-3*w-np.sqrt(3)*sm)*no*sc*v/np.sqrt(3)
    D = 12*sc**2 +vf*(O*v)**2-no*sc*v/np.sqrt(3)+4*(no**2)/3

    return (A+B+C)/D"""

# Differential equations
def NOE(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sp, sm, sc, sa, no]

    return (q(params_q) - 4 * sp) * no

def SPE(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sp, sm, sc, sa, no]
    params_q3 = [sc, no]

    Q = Q_three(params_q3)
    q_val = q(params_q)

    return (q_val - 2)*sp + (no**2)/3 + 0.25*Q*v - 3*sc**2

def SME(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sp, sm, sc, sa, no]
    params_q3 = [sc, no]

    Q = Q_three(params_q3)
    q_val = q(params_q)
    return (q_val - 2)*sm - np.sqrt(3)*(sc**2 - 2*sa**2 + 0.25*Q*v)

def SCE(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sp, sm, sc, sa, no]
    
    q_val = q(params_q)
    return (q_val - 2 + 3*sp + np.sqrt(3)*sm)*sc

def SAE(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sp, sm, sc, sa, no]
    
    q_val = q(params_q)

    return (q_val - 2 + 2*np.sqrt(3)*sm)*sa

def VE(params):
    sp, sm, sc, sa, no, v = params
    return v*(3*w - 1 - sp + np.sqrt(3)*sm)*(1 - v**2)/(1 - w*v**2)

def system(t, y):
    sp, sm, sc, sa, no, v = y
    return [
        SPE(y),
        SME(y),
        SCE(y),
        SAE(y),
        NOE(y),
        VE(y)
    ]

# Initial conditions
N1_max = 2*np.sqrt(3)
N1 = 0.1*N1_max
R = np.sqrt(1-O-(N1**2)/12)
print("Is the radius valid", R)
Angle1 = [np.arccos(-0.7)]
Angle2 = np.arccos(0.8)
Angle3 = 0

initial_conditions_list = []
for phi in Angle1:
    """SPI = R * np.sin(phi) * np.sin(Angle2) * np.cos(Angle3)
    SMI = R * np.sin(phi) * np.sin(Angle2) * np.sin(Angle3)
    SAI = R * np.sin(phi) * np.cos(Angle2)
    SCI = R * np.cos(phi)"""

    SMI = R*np.cos(phi)
    SPI = R*np.sin(phi)*np.cos(Angle2)
    SCI = R*np.sin(phi)*np.sin(Angle2)
    SAI=-0.00

    """SMI = -0.5
    SCI = 0.5
    SAI = 0
    #N1 = -4.654*10**(-6)
    N1=-0.05"""

    initial_free_conditions = [SPI, SMI, SCI, SAI, N1]
    
    #SPI = SP(initial_free_conditions)
    VI = tilted_velocity(initial_free_conditions)

    initial_conditions = [SPI, SMI, SCI, SAI, N1, VI]

    print('Initial tilt', VI)
    print('Initial SC: ', SCI)
    print('Initial NI: ', N1)
    print('Initial Q_3: ', -N1*SCI/np.sqrt(3))
    print('Initial Omega constant: ', O, 'initial function Omega: ', Omega_fun(initial_free_conditions))
    
    print()

    d1 = Constraint_1(initial_conditions)
    d2 = Constraint_2(initial_conditions)
    
    print('The initial difference in density equations: ', d1)
    print('The initial difference in Q_3: ', d2)
    print()
    initial_conditions_list.append(initial_conditions)

# Integration settings
tf = -9
resolution = 10**4
t_span = (0, tf)
t_eval = np.linspace(0, tf, resolution)

# Vector field grids
no_vals, sp_vals = np.meshgrid(np.linspace(-np.sqrt(12), np.sqrt(12), 25), np.linspace(-1, 1, 25))
sm_vals, sc_vals = np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))

# Precompute vector field values
NOE_of_SPE_vals = NOE([sp_vals, 0, 0, 0, no_vals, 0])
SPE_of_NOE_vals = SPE([sp_vals, 0, 0, 0, no_vals, 0])
SPE_of_SME_vals = SPE([sp_vals, sm_vals, 0, 0, 0, 0]) 
SME_of_SPE_vals = SME([sp_vals, sm_vals, 0, 0, 0, 0])
NOE_of_SME_vals = NOE([0, sm_vals, 0, 0, no_vals, 0])
SME_of_NOE_vals = SME([0, sm_vals, 0, 0, no_vals, 0])
NOE_of_SCE_vals = NOE([0, 0, sc_vals, 0, no_vals, 0])
SCE_of_NOE_vals = SCE([0, 0, sc_vals, 0, no_vals, 0])
SPE_of_SCE_vals = SPE([sp_vals, 0, sc_vals, 0, 0, 0])
SCE_of_SPE_vals = SCE([sp_vals, 0, sc_vals, 0, 0, 0])
SME_of_SCE_vals = SME([0, sm_vals, sc_vals, 0, 0, 0])
SCE_of_SME_vals = SCE([0, sm_vals, sc_vals, 0, 0, 0])

#Solve settings
max_step_1=10**(-6)
max_step_2=10**(-7)

abs_error = 10**(-15)
print("Absolute error tolerance: ", abs_error, " Max step size 1: ", max_step_1, " Max step size 2: ", max_step_2)
# Solve the system once for each initial condition and store the results
solutions1 = [solve_ivp(system, t_span, ic, t_eval=t_eval, events=time_limit_event, method='RK45', first_step=10**(-8), atol=abs_error, max_step=max_step_1) for ic in initial_conditions_list]
solutions2 = [solve_ivp(system, t_span, ic, t_eval=t_eval, events=time_limit_event, method='RK45', first_step=10**(-8), atol=abs_error, max_step=max_step_2) for ic in initial_conditions_list]

# Calculate the difference between the two solutions as a solution object
solutions = [
    OdeResult({
        't': solutions1.t,
        'y': np.abs(solutions2.y - solutions1.y),
        't_events': None,
        'y_events': None,
        'sol': None,
        'success': solutions1.success and solutions2.success,
        'message': 'Difference between solution2 and solution1'
    })
    for solutions1, solutions2 in zip(solutions1, solutions2)
]

print()
N = 1
# After solving the system, print the first value of each variable for each solution
"""for i, solution in enumerate(solutions):
    print(f"Initial values for solution {i + 1}:")
    print(f"sp: {solution.y[0][0]}")
    print(f"sm: {solution.y[1][0]}")
    print(f"sc: {solution.y[2][0]}")
    print(f"sa: {solution.y[3][0]}")
    print(f"no: {solution.y[4][0]}")
    print(f"v: {solution.y[5][0]}")
    print()
    print(f"Constraint on density evolution equations at step {N}: {Constraint_1(solution.y[N]0")
    print(f"Constraint on Q3 at step {N}: {Constraint_2(solution.y[N], solution.y[1][N], solution.y[2][N], solution.y[3][N], solution.y[4][N], solution.y[5][N])}")
    print()"""

# Create a figure with GridSpec to accommodate 5 subplots
fig = plt.figure(figsize=(14, 32))
gs = fig.add_gridspec(8, 2, height_ratios=[1, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])
ax6 = fig.add_subplot(gs[3, :])
ax7 = fig.add_subplot(gs[4, :])
ax8 = fig.add_subplot(gs[5, :])

# (no, sp) vector field in ax1
ax1.quiver(sp_vals, no_vals, SPE_of_NOE_vals, NOE_of_SPE_vals)
for i, solution in enumerate(solutions):
    sp_sol, no_sol = solution.y[0], solution.y[4]
    ax1.plot(sp_sol, no_sol, label=f"IC {i+1}")
ax1.set_title(r'$(\Sigma_+,0,0,N_1,0)\quad w=0, \quad N_1=\sqrt{3}$')
ax1.set_ylabel(r'$N_1$')
ax1.set_xlabel(r'$\Sigma_+$')
ax1.grid()

# (sp, sm) vector field in ax2
ax2.quiver(sp_vals, sm_vals, SPE_of_SME_vals, SME_of_SPE_vals)
for i, solution in enumerate(solutions):
    sp_sol, sm_sol = solution.y[0], solution.y[1]
    ax2.plot(sp_sol, sm_sol, label=f"IC {i+1}")
    ax2.text(sp_sol[0] + 0.05, sm_sol[0] + 0.02, f'IC {i+1}', fontsize=9,
             ha='right', color='cyan', backgroundcolor=(0.522, 0.192, 0.373, 0.369))
ax2.set_title(r'$(\Sigma_+, \Sigma_-,0,0,0),\quad w=0, \quad N_1=\sqrt{3}$')
ax2.set_xlabel(r'$\Sigma_+$')
ax2.set_ylabel(r'$\Sigma_-$')
ax2.grid()

# (sm, no) vector field in ax3
ax3.quiver(sm_vals, no_vals, SME_of_NOE_vals, NOE_of_SME_vals)
for i, solution in enumerate(solutions):
    sm_sol, no_sol = solution.y[1], solution.y[4]
    ax3.plot(sm_sol, no_sol, label=f"IC {i+1}")
ax3.set_title(r'$(0, \Sigma_-,0,N_1,0),\quad w=0, \quad N_1=\sqrt{3}$')
ax3.set_xlabel(r'$\Sigma_-$')
ax3.set_ylabel(r'$N_1$')
ax3.grid()

# (sc, no) vector field in ax4
ax4.quiver(sc_vals, no_vals, SCE_of_NOE_vals, NOE_of_SCE_vals)
for i, solution in enumerate(solutions):
    sc_sol, no_sol = solution.y[2], solution.y[4]
    ax4.plot(sc_sol, no_sol, label=f"IC {i+1}")
ax4.set_title(r'$(0,0,\Sigma_C,N_1,0),\quad w=0, \quad N_1=\sqrt{3}$')
ax4.set_xlabel(r'$\Sigma_C$')
ax4.set_ylabel(r'$N_1$')
ax4.grid()

# Fifth subplot: Time evolution of each solution component in ax5
# Use the first initial condition to define the colors for each component.
handles = {}
for i, solution in enumerate(solutions):
    if i == 0:
        line_sp, = ax5.plot(solution.t, solution.y[0], label="sp")
        line_sm, = ax5.plot(solution.t, solution.y[1], label="sm")
        line_sc, = ax5.plot(solution.t, solution.y[2], label="sc")
        line_sa, = ax5.plot(solution.t, solution.y[3], label="sa")
        line_no,  = ax5.plot(solution.t, solution.y[4], label="no")
        line_v,  = ax5.plot(solution.t, solution.y[5], label="v")
        handles = {"sp": line_sp, "sm": line_sm, "sc": line_sc, "sa":line_sa, "no": line_no, "v": line_v}
    else:
        ax5.plot(solution.t, solution.y[0], color=handles["sp"].get_color())
        ax5.plot(solution.t, solution.y[1], color=handles["sm"].get_color())
        ax5.plot(solution.t, solution.y[2], color=handles["sc"].get_color())
        ax5.plot(solution.t, solution.y[3], color=handles["sa"].get_color())
        ax5.plot(solution.t, solution.y[4], color=handles["no"].get_color())
        ax5.plot(solution.t, solution.y[5], color=handles["v"].get_color())
ax5.set_title("Solution Components vs Time")
ax5.set_xlabel("Time")
ax5.set_ylabel("Component values")
ax5.grid()
# Place the legend outside the plot on the right
ax5.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Sixth subplot: Omega_fun as a function of time in ax6
for i, solution in enumerate(solutions):
    Omega_values = Omega_fun(solution.y[0:5])
    log_Omega = np.log10(np.abs(Omega_values))  
    ax6.plot(solution.t, log_Omega, label=f"Density")
ax6.set_title("Log of Density vs Time")
ax6.set_xlabel("Time")
ax6.set_ylabel(r"Log$_{10}$ of Density")
ax6.grid()
ax6.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Seventh subplot: Difference of Q3 definitions
for i, solution in enumerate(solutions):
    O = Omega_fun(solution.y[0:5])
    v_tilt = tilted_velocity(solution.y[0:5])
    v_def = (1+w)/(1+w*v_tilt**2)*O*v_tilt
    
    Difference = np.abs(Q_three([solution.y[2], solution.y[4]]) - v_def)
    print(Difference)
    log_difference = np.log10(np.abs(Difference))
    ax7.plot(solution.t, log_difference, label=f"Q difference")
ax7.set_title("Log of Difference between Qs vs Time")
ax7.set_xlabel("Time")
ax7.set_ylabel(r"Difference in $Q_3$ definitions")
ax7.grid()
ax7.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Eigth subplot: Difference in the density evolution equations
for i, solution in enumerate(solutions):
    sp = solution.y[0]
    sm = solution.y[1]
    sc = solution.y[2]
    sa = solution.y[3]
    no = solution.y[4]
    v = solution.y[5]

    q_val = q([sp, sm, sc, sa, no])
    O = Omega_fun([sp, sm, sc, sa, no])

    OE1 = 2*(q_val-2)*O+3*(1-w)*O+(1-3*w-sp-np.sqrt(3)*sm)*(no*sc*v)*(1/np.sqrt(3))

    OE2P1 = 2*(2-q_val)*(1-O)-(1/3)*(1-4*sp)*no**2-2*(6*sp-np.sqrt(3)*sm)*sc**2
    OE2P2 = ((1+w)/(w+w*v**2))*(O**2)*(v**2)*(np.sqrt(3)*sm-sp)
    OE2 = OE2P1 + OE2P2

    Difference = np.abs(OE1 - OE2)
    log_difference = np.log10(np.abs(Difference))
    ax8.plot(solution.t, log_difference, label=f"Omega evolution difference")
ax8.set_title("Log of Difference between density equations")
ax8.set_xlabel("Time")
ax8.set_ylabel(r"Difference in $\Omega$ definitions")
ax8.grid()
ax8.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Display initial conditions at the top of the plot
text_str = "\n\n".join([f"IC {i+1}: SPI={ic[0]:.2f}, SMI={ic[1]:.2f}, SCI={ic[2]:.2f}, SAI={ic[3]:.2f}, N1={ic[4]:.2f}, VI={ic[5]:.2f}" 
                      for i, ic in enumerate(initial_conditions_list)])
fig.text(0.5, 0.95, text_str, fontsize=20, va='top', ha='center',
         bbox=dict(facecolor='lightgreen', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.9])

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()

#cProfile.run('solve_ivp(system, t_span, initial_conditions_list[0], t_eval=t_eval, method="BDF")')
