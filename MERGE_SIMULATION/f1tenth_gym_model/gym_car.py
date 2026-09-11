import numpy as np
from scipy.integrate import solve_ivp


#########################################################################################################
#########################################################################################################


class Car():
    """
    Simulates a 1:10 scale autonomous vehicle using a Single-Track (Bicycle) Model.
    
    The model supports two modes:
    1. Kinematic: Used at low speeds (< 0.5 m/s) to avoid singularities.
    2. Dynamic: Incorporates tire forces, slip angles, and longitudinal load transfer.
    """
    
    def __init__(self, dt:float, mu:float, C_Sf:float, C_Sr:float, lf:float, lr:float, 
                 h:float, m:float, I:float, s_min:float, s_max:float, sv_min:float, 
                 sv_max:float, v_switch:float, a_max:float, v_min:float, v_max:float, 
                 width:float, length:float, g:float, X0:np.ndarray
                ):
        """
        Initializes vehicle parameters.

        Args:
            dt: Simulation time step (s).
            mu: Friction coefficient between tires and track.
            C_Sf/C_Sr: Cornering stiffness coefficients for front and rear tires.
            lf/lr: Distance from Center of Mass (CoM) to front and rear axles (m).
            h: Height of the Center of Mass (m).
            m: Total vehicle mass (kg).
            I: Yaw moment of inertia (kg*m^2).
            s_min/s_max: Minimum and maximum steering angles (rad).
            sv_min/sv_max: Minimum and maximum steering velocities (rad/s).
            v_switch: Velocity threshold for motor power limits (m/s).
            a_max: Maximum longitudinal acceleration (m/s^2).
            v_min/v_max: Global velocity bounds (m/s).
            width/length: Physical dimensions of the vehicle (m).
            g: Gravitational acceleration (m/s^2).
            X0: Initial state vector [x, y, delta, v, psi, yaw_rate, beta].
        """
        self.mu = mu
        self.C_Sf = C_Sf
        self.C_Sr = C_Sr
        self.lf = lf
        self.lr = lr
        self.h = h
        self.m = m
        self.I = I
        self.s_min = s_min
        self.s_max = s_max
        self.sv_min = sv_min
        self.sv_max = sv_max
        self.v_switch = v_switch
        self.a_max = a_max
        self.v_min = v_min
        self.v_max = v_max
        self.width = width
        self.length = length
        self.g = g
        self.dt = dt
        self.reset(X0)


    def reset(self, X0:np.ndarray, kinematic_flag:bool=False):
        """
        Resets the vehicle state and simulation clock.

        Args:
            X0: New initial state vector.
            kinematic_flag: If True, forces the model to stay in the kinematic regime.
        """
        self.kinematic_flag = kinematic_flag
        self.X = np.copy(X0)
        self.t = 0.

    
    def dynamic_ODE(self, t, X):
        """
        Defines the system of differential equations (dot{X} = f(X, U)).

        Args:
            t: Current simulation time (internal use for solve_ivp).
            X: Current state vector:
                X[0], X[1]: Global position (x, y)
                X[2]: Steering angle (delta)
                X[3]: Velocity magnitude (v)
                X[4]: Yaw angle (psi)
                X[5]: Yaw rate (psi_dot)
                X[6]: Slip angle at vehicle center (beta)

        Returns:
            f: The derivative vector (dot{X}).
        """

        steering_velocity, acceleration = self.U
        
        # 1. Apply Motor Power Limits (Power = Force * Velocity)
        # At high speeds, the available acceleration is limited by constant power.
        pos_acc_limit = self.a_max * self.v_switch / X[3] if X[3] > self.v_switch else self.a_max
        
        # Apply global velocity bounds and physical acceleration limits
        if (X[3] <= self.v_min and acceleration <= 0) or (X[3] >= self.v_max and acceleration >= 0): 
            acceleration = 0.
        elif acceleration <= -self.a_max: 
            acceleration = -self.a_max
        elif acceleration >= pos_acc_limit: 
            acceleration = pos_acc_limit

        # 2. Apply Steering Servo Limits (Mechanical stops and slew rate)
        if (X[2] <= self.s_min and steering_velocity <= 0) or (X[2] >= self.s_max and steering_velocity >= 0): 
            steering_velocity = 0.
        elif steering_velocity <= self.sv_min: 
            steering_velocity = self.sv_min
        elif steering_velocity >= self.sv_max: 
            steering_velocity = self.sv_max

        U = np.array([steering_velocity, acceleration])
        
        # 3. Model Regime Selection
        # Use Kinematic Model if speed is very low (prevents div-by-zero in slip calculations)
        if abs(X[3]) < 0.5 or self.kinematic_flag:
            f = np.array([
                X[3] * np.cos(X[4]),                         # x_dot
                X[3] * np.sin(X[4]),                         # y_dot
                U[0],                                        # delta_dot
                U[1],                                        # v_dot
                X[3] / (self.lr + self.lf) * np.tan(X[2]),   # psi_dot (yaw rate)
                0.,                                          # Dynamic states are inactive
                0.,
            ])

        else:
            # 4. Single-Track Dynamic Model
            # Incorporates lateral slip and yaw momentum.
            # Dynamic equations account for longitudinal load transfer (h * acc) 
            # which affects normal force and thus cornering stiffness.
            
            f = np.array([
                X[3] * np.cos(X[6] + X[4]),                  # x_dot (Velocity vector includes slip beta)
                X[3] * np.sin(X[6] + X[4]),                  # y_dot
                U[0],                                        # delta_dot
                U[1],                                        # v_dot
                X[5],                                        # psi_dot = yaw_rate
                
                # Yaw Acceleration (psi_double_dot): Derived from sum of lateral tire moments
                (-self.mu*self.m/(X[3]*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(self.g*self.lr-U[1]*self.h) + 
                self.lr**2*self.C_Sr*(self.g*self.lf + U[1]*self.h))*X[5] +
                self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.g*self.lf + U[1]*self.h) - self.lf*self.C_Sf*(self.g*self.lr - U[1]*self.h))*X[6] +
                self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(self.g*self.lr - U[1]*self.h)*X[2]),

                # Side Slip Rate (beta_dot): Derived from lateral force balance
                ((self.mu/(X[3]**2*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + U[1]*self.h)*self.lr - self.C_Sf*(self.g*self.lr - U[1]*self.h)*self.lf)-1)*X[5] - 
                 self.mu/(X[3]*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + U[1]*self.h) + self.C_Sf*(self.g*self.lr-U[1]*self.h))*X[6] +
                 self.mu/(X[3]*(self.lr+self.lf))*(self.C_Sf*(self.g*self.lr-U[1]*self.h))*X[2])
            ])
        return f
    

    def step_sim(self, U:np.ndarray):
        """
        Advances the simulation by one time step using an IVP solver.

        Args:
            U: Control input vector [steering_velocity, acceleration].
        """
        self.U = U
        # Integrate the ODE from current time t to t + dt
        self.X = solve_ivp(
            self.dynamic_ODE, 
            [self.t, self.t + self.dt], 
            np.copy(self.X),
            # Tight tolerance for high-speed stability
            atol=1e-7,
        ).y[:, -1] # Extract the state at the final time point
        self.t += self.dt


#########################################################################################################
#########################################################################################################