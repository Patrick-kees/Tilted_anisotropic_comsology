import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import cProfile

# Constants
w = 1/3

# Start the timer for the event
start_time = time.time()

def time_limit_event(t, y):
    # Stop after 600 seconds (10 minutes)
    if time.time() - start_time > 6000:
        return 0
    return 1

time_limit_event.terminal = True

# Model functions
def Q_three(sc, no):
    return -(no * sc) / np.sqrt(3)

def Omega_fun(sp, sm, sc, no):
    return 1 - (sp**2 + sm**2 + sc**2) - (no**2) / 12

def tilted_velocity(sp, sm, sc, no):
    Q = Q_three(sc, no)
    O = Omega_fun(sp, sm, sc, no)
    A = (1 + w) * O
    B = ((1 + w)**2) * O**2
    C = 4 * w * (Q)**2
    return (2 * Q) / (A + np.sqrt(np.abs(B - C)))

def q(sp, sm, sc, no):
    Q = Q_three(sc, no)
    O = Omega_fun(sp, sm, sc, no)
    v_tilt = tilted_velocity(sp, sm, sc, no)
    SigmaSquared = sp**2 + sm**2 + sc**2
    return 2 * SigmaSquared + 0.5 * (1 + 3*w) * O + 0.5 * (1 - 3*w) * Q * v_tilt

# Differential equations
def NOE(sp, sm, sc, no, v):
    return (q(sp, sm, sc, no) - 4 * sp) * no

def SPE(sp, sm, sc, no, v):
    Q = Q_three(sc, no)
    return (q(sp, sm, sc, no) - 2)*sp + (no**2)/3 + 0.25*Q*tilted_velocity(sp, sm, sc, no) - 3*sc**2

def SME(sp, sm, sc, no, v):
    Q = Q_three(sc, no)
    return (q(sp, sm, sc, no) - 2)*sm - np.sqrt(3)*(sc**2 + 0.25*Q*tilted_velocity(sp, sm, sc, no))

def SCE(sp, sm, sc, no, v):
    return (q(sp, sm, sc, no) - 2 + 3*sp + np.sqrt(3)*sm)*sc

def VE(sp, sm, sc, no, v):
    v_tilt = tilted_velocity(sp, sm, sc, no)
    return v_tilt*(3*w - 1 - sp + np.sqrt(3)*sm)*(1 - v_tilt**2)/(1 - w*v_tilt**2)

def system(t, y):
    sp, sm, sc, no, v = y
    return [
        SPE(sp, sm, sc, no, v),
        SME(sp, sm, sc, no, v),
        SCE(sp, sm, sc, no, v),
        NOE(sp, sm, sc, no, v),
        VE(sp, sm, sc, no, v)
    ]

# Initial conditions
N1 = 0.7*np.sqrt(3)
RF=0.75
R = RF*np.sqrt(1 - (N1**2) / 12)
theta = np.arccos(1/4)
phi_angles = [-3*np.pi/8]

initial_conditions_list = []
for phi in phi_angles:
    SPI = R * np.sin(theta) * np.cos(phi)
    SMI = R * np.sin(theta) * np.sin(phi)
    SCI = R * np.cos(theta)
    VI = tilted_velocity(SPI, SMI, SCI, N1)
    
    initial_conditions_list.append([SPI, SMI, SCI, N1, VI])

# Integration settings
tf = -100
t_span = (0, tf)
t_eval = np.linspace(0, tf, 500000)

# Vector field grids
no_vals, sp_vals = np.meshgrid(np.linspace(0, np.sqrt(12), 25), np.linspace(-1, 1, 25))
sm_vals, sc_vals = np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))

# Precompute vector field values
NOE_of_SPE_vals = NOE(sp_vals, 0, 0, no_vals, 0)
SPE_of_NOE_vals = SPE(sp_vals, 0, 0, no_vals, 0)
SPE_of_SME_vals = SPE(sp_vals, sm_vals, 0, 0, 0)
SME_of_SPE_vals = SME(sp_vals, sm_vals, 0, 0, 0)
NOE_of_SME_vals = NOE(0, sm_vals, 0, no_vals, 0)
SME_of_NOE_vals = SME(0, sm_vals, 0, no_vals, 0)
NOE_of_SCE_vals = NOE(0, 0, sc_vals, no_vals, 0)
SCE_of_NOE_vals = SCE(0, 0, sc_vals, no_vals, 0)
SPE_of_SCE_vals = SPE(sp_vals, 0, sc_vals, 0, 0)
SCE_of_SPE_vals = SCE(sp_vals, 0, sc_vals, 0, 0)
SME_of_SCE_vals = SME(0, sm_vals, sc_vals, 0, 0)
SCE_of_SME_vals = SCE(0, sm_vals, sc_vals, 0, 0)

# Solve the system once for each initial condition and store the results
solutions = [solve_ivp(system, t_span, ic, t_eval=t_eval, events=time_limit_event, method='LSODA') for ic in initial_conditions_list]

# Create a figure with GridSpec to accommodate 5 subplots
fig = plt.figure(figsize=(14, 32))
gs = fig.add_gridspec(7, 2, height_ratios=[1, 1, 0.8, 0.8, 0.8, 0.8, 0.8])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])
ax6 = fig.add_subplot(gs[3, :])
ax7 = fig.add_subplot(gs[4, :])

# (no, sp) vector field in ax1
ax1.quiver(sp_vals, no_vals, SPE_of_NOE_vals, NOE_of_SPE_vals)
for i, solution in enumerate(solutions):
    sp_sol, no_sol = solution.y[0], solution.y[3]
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
    sm_sol, no_sol = solution.y[1], solution.y[3]
    ax3.plot(sm_sol, no_sol, label=f"IC {i+1}")
ax3.set_title(r'$(0, \Sigma_-,0,N_1,0),\quad w=0, \quad N_1=\sqrt{3}$')
ax3.set_xlabel(r'$\Sigma_-$')
ax3.set_ylabel(r'$N_1$')
ax3.grid()

# (sc, no) vector field in ax4
ax4.quiver(sc_vals, no_vals, SCE_of_NOE_vals, NOE_of_SCE_vals)
for i, solution in enumerate(solutions):
    sc_sol, no_sol = solution.y[2], solution.y[3]
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
        line_no, = ax5.plot(solution.t, solution.y[3], label="no")
        line_v,  = ax5.plot(solution.t, solution.y[4], label="v")
        handles = {"sp": line_sp, "sm": line_sm, "sc": line_sc, "no": line_no, "v": line_v}
    else:
        ax5.plot(solution.t, solution.y[0], color=handles["sp"].get_color())
        ax5.plot(solution.t, solution.y[1], color=handles["sm"].get_color())
        ax5.plot(solution.t, solution.y[2], color=handles["sc"].get_color())
        ax5.plot(solution.t, solution.y[3], color=handles["no"].get_color())
        ax5.plot(solution.t, solution.y[4], color=handles["v"].get_color())
ax5.set_title("Solution Components vs Time")
ax5.set_xlabel("Time")
ax5.set_ylabel("Component values")
ax5.grid()
# Place the legend outside the plot on the right
ax5.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Sixth subplot: Omega_fun as a function of time in ax6
for i, solution in enumerate(solutions):
    Omega_values = Omega_fun(solution.y[0], solution.y[1], solution.y[2], solution.y[3])
    log_Omega = np.log10(np.abs(Omega_values))  
    ax6.plot(solution.t, log_Omega, label=f"Density")
ax6.set_title("Log of Density vs Time")
ax6.set_xlabel("Time")
ax6.set_ylabel(r"Log$_{10}$ of Density")
ax6.grid()
ax6.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Seventh subplot: Difference of Q3 definitions
for i, solution in enumerate(solutions):
    v_tilt = tilted_velocity(solution.y[0], solution.y[1], solution.y[2], solution.y[3])
    O = Omega_fun(solution.y[0], solution.y[1], solution.y[2], solution.y[3])
    v_def = (1+w)/(1+w*v_tilt**2)*O*v_tilt

    Difference = np.abs(Q_three(solution.y[2], solution.y[3]) - v_def)
    log_difference = np.log10(Difference)
    ax7.plot(solution.t, log_difference, label=f"Q difference")
ax7.set_title("Log of Difference between Qs vs Time")
ax7.set_xlabel("Time")
ax7.set_ylabel(r"Difference in $Q_3$ definitions")
ax7.grid()
ax7.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Display initial conditions at the top of the plot
text_str = "\n\n".join([f"IC {i+1}: SPI={ic[0]:.2f}, SMI={ic[1]:.2f}, SCI={ic[2]:.2f}, N1={ic[3]:.2f}, VI={ic[4]:.2f}" 
                      for i, ic in enumerate(initial_conditions_list)])
fig.text(0.5, 0.95, text_str, fontsize=20, va='top', ha='center',
         bbox=dict(facecolor='lightgreen', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()

#cProfile.run('solve_ivp(system, t_span, initial_conditions_list[0], t_eval=t_eval, method="BDF")')
