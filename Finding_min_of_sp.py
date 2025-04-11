import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize
import time

# Constants
w = 1/3 #speed of sound sqaured
O = 0.025 #Hubble normalized density of the universe

# Start the timer for the event
start_time = time.time()

def time_limit_event(t, y):
    # Stop after 600 seconds (10 minutes)
    if time.time() - start_time > 600:
        return 0
    return 1

time_limit_event.terminal = True

# Model functions
def Q_three(params):
    sc, no = params
    return -(no * sc) / np.sqrt(3)

def Omega_fun(sp, sm, sc, sa, no):
    return 1 - (sp**2 + sm**2 + sc**2 +sa**2) - (no**2) / 12

def tilted_velocity(params):
    sm, sc, sa, no = params
    params_q3 = [sc, no]

    Q = Q_three(params_q3)

    A = (1 + w) * O
    B = ((1 + w)**2) * O**2
    C = 4 * w * (Q)**2

    tilted_velocity = (2 * Q) / (A + np.sqrt(np.abs(B - C)))
    return tilted_velocity

def q(params):
    sm, sc, sa, no = params
    params_q3 = [sc, no]
    params_v = [sm, sc, sa, no]

    Q = Q_three(params_q3)
    v_tilt = tilted_velocity(params_v)
    SigmaSquared = 1-O-(no**2)/12

    q_val = 2 * SigmaSquared + 0.5 * (1 + 3*w) * O + 0.5 * (1 - 3*w) * Q * v_tilt
    return q_val

#Constraints
    #Density evolution equation difference
def Constraint_1(sp, sm, sc, sa, no, v):
    q_val = q(sp, sc, sa, no)

    OE1 = 2*(q_val-2)*O+3*(1-w)*O+(1-3*w-sp-np.sqrt(3)*sm)*(no*sc*v)*(1/np.sqrt(3))

    OE2P1 = 2*(2-q_val)*(1-O)-(1/3)*(1-4*sp)*no**2-2*(6*sp-np.sqrt(3)*sm)*sc**2
    OE2P2 = ((1+w)/(w+w*v**2))*(O**2)*(v**2)*(np.sqrt(3)*sm-sp)
    OE2 = OE2P1 + OE2P2

    Difference = OE1 - OE2

    return Difference
    #Q_3 equation difference
def Constraint_2(sp, sm, sc, sa, no ,v):
    v_def = (1+w)/(1+w*v**2)*O*v

    Difference = np.abs(Q_three(sc,no) - v_def)

    return Difference

#SM function (rearranged Constraint_1)
def SM(sp, sc, sa, no, v):
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

    return (A+B+C)/D

# Differential equations
def NOE(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sm, sc, sa, no]

    return (q(params_q) - 4 * sp) * no

def SPE(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sm, sc, sa, no]
    params_q3 = [sc, no]

    Q = Q_three(params_q3)
    q_val = q(params_q)

    return (q_val - 2)*sp + (no**2)/3 + 0.25*Q*v - 3*sc**2

def SME(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sm, sc, sa, no]
    params_q3 = [sc, no]

    Q = Q_three(params_q3)
    q_val = q(params_q)
    return (q_val - 2)*sm - np.sqrt(3)*(sc**2 - 2*sa**2 + 0.25*Q*v)

def SCE(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sm, sc, sa, no]
    
    q_val = q(params_q)
    return (q_val - 2 + 3*sp + np.sqrt(3)*sm)*sc

def SAE(params):
    sp, sm, sc, sa, no, v = params
    params_q = [sm, sc, sa, no]
    
    q_val = q(params_q)

    return (q_val - 2 + 2*np.sqrt(3)*sm)*sa

def VE(params):
    sp, sm, sc, sa, no, v = params
    return v*(3*w - 1 - sp + np.sqrt(3)*sm)*(1 - v**2)/(1 - w*v**2)

def system(t, y):
    sp, sm, sc, sa, no, v = y
    return [
        SPE(sp, sm, sc, sa, no, v),
        SME(sp, sm, sc, sa, no, v),
        SCE(sp, sm, sc, sa, no, v),
        SAE(sp, sm, sc, sa, no, v),
        NOE(sp, sm, sc, sa, no, v),
        VE(sp, sm, sc, sa, no, v)
    ]

# Initial conditions
N1_max = 2*np.sqrt(3)
N1 = 0.1*N1_max
R = np.sqrt(1-O-(N1**2)/12)
print("Is the radius valid", R)
Angle1 = [-1*np.pi/8]
Angle2 = np.arccos(-0.8)
Angle3 = 0

initial_conditions_list = []
for phi in Angle1:
    """SPI = R * np.sin(phi) * np.sin(Angle2) * np.cos(Angle3)
    SMI = R * np.sin(phi) * np.sin(Angle2) * np.sin(Angle3)
    SAI = R * np.sin(phi) * np.cos(Angle2)
    SCI = R * np.cos(phi)"""
    
    """SMI = R*np.cos(Angle2)
    SAI = 0.00
    SCI = R*np.sin(Angle2)*np.cos(Angle3)"""

    SMI = -0.5
    SCI = 0.5
    SAI = 0
    N1 = -4.654*10**(-6)

    initial_free_conditions = [SMI, SCI, SAI, N1]
    VI = tilted_velocity(initial_free_conditions)
    initial_free_conditions_VI = [SMI, SCI, SAI, N1]
    SPI = SP(initial_free_conditions_VI)

    print ('Here is the initial value of SP: ', SPI)
    print('Here is initial tilt', VI)
    print('Here is Initial SC: ', SCI)
    print('Here is Initial NI: ', N1)
    print('Here is Initial Q_3: ', -N1*SCI/np.sqrt(3))
    
    print()

    #d1 = Constraint_1(SPI, SMI, SCI, SAI, N1, VI)
    #d2 = Constraint_2(SPI, SMI, SCI, SAI, N1, VI)
    
    #print('Here is the difference in density equations: ', d1)
    #print('Here is the difference in Q_3: ', d2)

    initial_conditions_list_minus_sp = [SMI, SCI, SAI, N1]
    bounds = [(-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-np.sqrt(12),np.sqrt(12))]

    min = optimize.minimize(SP,initial_conditions_list_minus_sp, bounds=bounds)
    print('Here is the minimum of SP: ', min)