#%%
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

h =  sc.constants.Planck
e = sc.constants.elementary_charge
phi_0 = h/(2*e)
mu_0 = sc.constants.mu_0
eps_0 = sc.constants.epsilon_0
mu0 = sc.constants.mu_0
delta_0 = 176e-6 # superconducting gap of Al for T << Tc
eps_r_AlOx = 9

#%% Functions 
def K_func(w, s, H):
    k = w / (w+2*s)
    K = sc.special.ellipk(k)
    return K

def Kp_func(w, s, H):
    k = w / (w+2*s)
    kp = np.sqrt(1-k**2)
    Kp = sc.special.ellipk(kp)
    return Kp

def K1_func(w, s, H):
    b = w + 2*s
    k1 = np.tanh(np.pi*w/(4*H)) / np.tanh(np.pi*b/(4*H))
    K1 = sc.special.ellipk(k1)
    return K1

def K1p_func(w, s, H):
    b = w + 2*s
    k1 = np.tanh(np.pi*w/(4*H)) / np.tanh(np.pi*b/(4*H))
    k1p = np.sqrt(1-k1**2)
    K1p = sc.special.ellipk(k1p)
    return K1p

#%% Class definitions

'''
Source for material properties: https://uspas.fnal.gov/materials/15Rutgers/1_SRF_Fundamentals.pdf
'''

class Junction:
    """
    Class describing an Josephson junction Al-AlOx-Al.

    Args:
        d_up: Thickness of the top Al layer.
        d_down: Thickness of the bottom Al layer.
        t: Thickness of the tunneling barrier.
        w: Width of the junction.

    """
    def __init__(self, d_up, d_down, t, w):
        self.d_up = d_up
        self.d_down = d_down
        self.t = t
        self.w = w
        self.london = 16e-9
        self.pippard = 1600e-9
        self.B_crit_bulk = 10e-3

        self.london_eff_up = self.london * np.sqrt(self.pippard/self.d_up)
        

        self.london_eff_down = self.london * np.sqrt(self.pippard/self.d_down)

        self.B_crit_in_up = self.B_crit_bulk * self.london_eff_up / self.d_up * np.sqrt(24)
        self.B_crit_in_down = self.B_crit_bulk * self.london_eff_down / self.d_down * np.sqrt(24)
        self.B_crit_in = min(self.B_crit_in_up, self.B_crit_in_down)

        self.B_crit_out = 1.65*phi_0/(w)**2

        A = self.w * (self.t + min(self.london_eff_up, self.d_up) + min(self.london_eff_down, self.d_down))
        self.B_phi0 = phi_0 / A

    def f_B_in(self, B_in, f_max):
        return f_max * np.power(1 - (B_in/self.B_crit_in)**2, 1./4) * np.sqrt(np.abs(np.sinc(B_in / self.B_phi0)))

    def plot_f_B_in(self, f_max=10e9):
        B_range = np.linspace(-self.B_crit_in, self.B_crit_in, 1000)
        plt.figure()
        plt.plot(B_range*1e3, self.f_B_in(B_range, f_max)*1e-9)
        plt.xlabel("$B_{\parallel}$ (mT)")
        plt.ylabel("f (GHz)")
        plt.plot()

    def __str__(self):
        repr = ""
        repr += "Geometrical parameters\n"
        repr += f"Al thickness top: {self.d_up*1e9:.0f} nm\n"
        repr += f"Al thickness bottom: {self.d_down*1e9:.0f} nm\n"
        repr += f"Oxide thickness: {self.t*1e9:.0f} nm\n"
        repr += f"Junction width: {self.w*1e9:.0f} nm\n"
        repr += "\nSuperconductor characteristic quantities\n"
        repr += f"London eff. length top: {self.london_eff_up*1e9:.0f} nm\n"
        repr += f"London eff. length bottom: {self.london_eff_down*1e9:.0f} nm\n"
        repr += f"In-plane Bcrit top: {self.B_crit_in_up*1e3:.1f} mT\n"
        repr += f"In-plane Bcrit bottom: {self.B_crit_in_down*1e3:.1f} mT\n"
        repr += f"In-plane one flux quantum B field: {self.B_phi0*1e3:.1f} mT\n"
        return repr

class SuperconductingFilm:
    def __init__(self, d, material):
        self.d = d
        self.material = material
        if material == "Al":
            self.london = 16e-9
            self.pippard = 1600e-9
            self.B_crit_bulk = 10e-3
        elif material == "Nb":
            self.london = 32e-9
            self.pippard = 39e-9
            self.B_crit_bulk = 100e-3
        self.london_eff = self.london * np.sqrt(self.pippard/self.d)
        self.B_crit_in = self.B_crit_bulk * self.london_eff / self.d * np.sqrt(24)

    def __str__(self):
        repr = ""
        repr += f"Material: {self.material}\n"
        repr += f"Film thickness: {self.d*1e9:.0f} nm\n"
        repr += f"London eff. length: {self.london_eff*1e9:.0f} nm\n"
        repr += f"In-plane Bcrit: {self.B_crit_in*1e3:.1f} mT\n"
        repr += f"Out-of-plane Bcrit: {self.B_crit_bulk*1e3:.1f} mT\n"
        return repr
    
class CPW:
    def __init__(self, w, s, t, l, H, eps_r = 11.9, type="lambda_half"):
        """
        Class for a CPW resonator.

        Args: 
            w: Width of the inner conductor.
            s: Spacing from the ground plane.
            t: Thickness of the metal.
            l: Length of the CPW.
            H: Height of the substrate.
            eps_r: Dielectric constant of the substrate (default value is 11.9).
            type:"lambda_half" (default) or "lambda_quarter".
    
        """
        self.w = w
        self.s = s
        self.t = t
        self.length = l
        self.H = H
        self.eps_r = eps_r
        self.type = type

        if type != "lambda_half" and type != "lambda_quarter":
            print('Error: the provided type must be either  "lambda_half" or  "lambda_quarter"')
            raise ValueError

        self.update()

    def update(self):
        # Elliptic integrals
        K = K_func(self.w, self.s, self.H)
        Kp = Kp_func(self.w, self.s, self.H)
        K1 = K1_func(self.w, self.s, self.H)
        K1p = K1p_func(self.w, self.s, self.H)
        Gamma = 1 / (K/Kp + K1/K1p)
        A = 2*Gamma
        B = 1/A
        eps_eff = (1 + self.eps_r * Kp/K*K1/K1p) / (1 + Kp/K*K1/K1p)
        # Capacitance and inductance per unit length
        self.C = 4*eps_eff*eps_0 * B
        self.L = mu0/4 * A

        self.Z0 = np.sqrt(self.L/self.C)
        self.vph = 1/np.sqrt(self.L*self.C)
        if self.type ==  "lambda_half":
            self.fr = self.vph/(2*self.length)
        elif self.type ==  "lambda_quarter":
            self.fr = self.vph/(4*self.length)

    def __repr__(self):
        repr = ""
        repr += f"Inductance per unit length L = {self.L*1e3:.2f} nH/um\n"
        repr += f"Capacitance per unit length C = {self.C*1e9:.2f} fF/um\n"
        repr += f"Phase velocity vph = {self.vph:.0f} m/s\n"
        repr += f"Characteristic impedance Z0 = {self.Z0:.0f} Ohm\n"
        repr += f"Resonance frequency fr = {self.fr*1e-9:.4f} GHz\n"

class JJ_array:
    def __init__(self, junction, w, s, H, l_junction, l_spurious, l_unit, N, RA, eps_r=11.9):
        """
        Class describing a JJ array resonator (so far only lambda/4).

        Args:
            junction: Junction object for the junctions in the resonators
            w: Width of the resonator
            s: Spacing between resonator core and ground plane
            H: Substrate thickness
            l_unit: Junction length
            l_spurious: Length of the spurious junction
            l_unit: Length of the junction unit
            N: Number of junctions
            RA: Junction resistance at room T times area
            eps_r: Dielectric constant of the substrate (default is Si)
        
        """
        self.J = junction
        self.w = w
        self.s = s
        self.H = H
        self.l_junction = l_junction
        self.l_spurious = l_spurious
        self.l_unit = l_unit
        self.N = N
        self.RA = RA
        self.eps_r = eps_r

        self.update()

    def update(self):
        """
        Recalculate all the resonator's parameters. You can use this function after changing one of the object's attributes.

        Returns: None
        
        """
    
        self.length = self.l_unit * self.N

        self.A_junction = self.w*self.l_junction
        self.A_spurious = self.w*self.l_spurious
        self.Ic0_junction = (delta_0*np.pi)/(2*self.RA/self.A_junction)
        self.Ic0_spurious = (delta_0*np.pi)/(2*self.RA/self.A_spurious)

        self.L_junction = phi_0/(2*np.pi*self.Ic0_junction)
        self.L_spurious = phi_0/(2*np.pi*self.Ic0_spurious)
        self.Ltot = (self.L_junction+self.L_spurious)*self.N
        self.L = self.Ltot/self.length
        self.n_sq = int(self.length/self.w)
        self.Lsq = self.Ltot/self.n_sq

        # Elliptic integrals
        K = K_func(self.w, self.s, self.H)
        Kp = Kp_func(self.w, self.s, self.H)
        K1 = K1_func(self.w, self.s, self.H)
        K1p = K1p_func(self.w, self.s, self.H)
        Gamma = 1 / (K/Kp + K1/K1p)
        A = 2*Gamma
        B = 1/A
        eps_eff = (1 + self.eps_r * Kp/K*K1/K1p) / (1 + Kp/K*K1/K1p)
        self.C = 4*eps_eff*eps_0 * B

        self.Z0 = np.sqrt(self.L/self.C)
        self.vph = 1/np.sqrt(self.L*self.C)
        self.fr = self.vph/(4*self.length)

        self.Leq = self.Ltot*(8/np.pi**2) 
        self.Ceq = self.C*self.length/2 
        self.Zeq = 4/np.pi * self.Z0
        self.fr_eq = 1/(2*np.pi*np.sqrt(self.Leq*self.Ceq))

        
        self.C_J = eps_0*eps_r_AlOx*self.w*self.l_junction/self.J.t

        self.Ec = e**2/(2*self.C_J)
        self.EJ = (phi_0/(2*np.pi))**2 /self.L_junction

        self.RJ = self.RA/(self.w * self.l_junction)
        self.alpha0 = h/(4*e**2*self.RJ)

        self.f_plasma = 1/np.sqrt(2*np.pi*self.L_junction*self.C_J)
    
    def fr_of_B_in(self, B_in):
        """
        Calculate the resonance frequency of the resonator as a function of the external in-plane B field.

        Args: 
            B_in: Array of the in-plane magnetic field.

        Returns: 
            Array with the resonance frequency.
        """
        return self.fr * np.power(1 - (B_in/self.J.B_crit_in)**2, 1./4) * np.sqrt(np.abs(np.sinc(B_in / self.J.B_phi0)))
    
    def fr_of_B_out(self, B_in):
        """
        Calculate the resonance frequency of the resonator as a function of the external out-of-plane B field.

        Args: 
            B_out: Array of the out-of-plane magnetic field.

        Returns: 
            Array with the resonance frequency.
        """
        return self.fr * np.power(1 - (B_in/self.J.B_crit_in)**2, 1./4)
    
    def plot_fr_of_B_in(self):
        """
        Plot the resonance frequency as a function of an external in-plane magnetic field.
        """
        B_range = np.linspace(-self.J.B_crit_in, self.J.B_crit_in, 1000)
        plt.figure()
        plt.plot(B_range*1e3, self.fr_of_B_in(B_range)*1e-9)
        plt.xlabel("$B_{\parallel}$ (mT)")
        plt.ylabel("f (GHz)")
        plt.plot()

    def plot_fr_of_B_out(self):
        """
        Plot the resonance frequency as a function of an external out-of-plane magnetic field.
        """
        B_range = np.linspace(-self.J.B_crit_out, self.J.B_crit_out, 1000)
        plt.figure()
        plt.plot(B_range*1e3, self.fr_of_B_out(B_range)*1e-9)
        plt.xlabel("$B_{\parallel}$ (mT)")
        plt.ylabel("f (GHz)")
        plt.plot()

    def __str__(self):
        repr = ""
        repr += "-------------------------------------\n"
        repr += "Junction design parameter\n"
        repr += "-------------------------------------\n"
        repr += f"Top Al thickness d_top = {self.J.d_up*1e9:.1f} nm\n"
        repr += f"Bottom Al thickness d_bottom = {self.J.d_down*1e9:.1f} nm\n"
        repr += f"Oxide thickness t = {self.J.t*1e9:.1f} nm\n"
        repr += f"Width w = {self.w*1e9:.1f} nm\n"
        repr += f"Junction length l_junction = {self.l_junction*1e9:.1f} nm\n"
        repr += f"Spurious junction length l_spurious = {self.l_spurious*1e9:.1f} nm\n"
        repr += "-------------------------------------\n"
        repr += "Junction physical quantities\n"
        repr += "-------------------------------------\n"
        repr += f"Junction inductance L_junction = {self.L_junction*1e9:.2f} nH/junction\n"
        repr += f"Spurious junction inductance L_spurious = {self.L_spurious*1e9:.2f} nH/junction\n"
        repr += f"Junction capacitance C_junction = {self.C_J*1e15:.2f} fF\n"
        repr += f"Ratio EJ/Ec = {self.EJ/self.Ec:.2f}\n"
        repr += f"Ratio alpha0 = {self.alpha0:.2f}\n"
        repr += "-------------------------------------\n"
        repr += "Behaviour in magnetic field\n"
        repr += "-------------------------------------\n"
        repr += f"In-plane Bcrit: {min(self.J.B_crit_in_up, self.J.B_crit_in_down)*1e3:.1f} mT\n"
        repr += f"Out-of-plane Bcrit: {self.J.B_crit_out*1e3:.1f} mT\n"
        repr += f"In-plane one flux quantum B field: {self.J.B_phi0*1e3:.1f} mT\n"
        repr += "-------------------------------------\n"
        repr += "Resonator parameters\n"
        repr += "-------------------------------------\n"
        repr += f"Number of junctions N = {self.N}\n"
        repr += f"Spacing between junctions l_spurious = {self.l_spurious*1e6:.1f} um\n"
        repr += f"Length l = {self.length*1e6:.2f} um\n"
        repr += f"Total inductance Ltot = {self.Ltot*1e9:.2f} nH\n"
        repr += f"Number of squares: {self.n_sq}\n"
        repr += f"Inductance per square Lsq = {self.Lsq*1e12:.0f} pH/sq\n"
        repr += f"Inductance per unit length L = {self.L*1e3:.2f} nH/um\n"
        repr += f"Capacitance per unit length C = {self.C*1e9:.2f} fF/um\n"
        repr += f"Phase velocity vph = {self.vph:.0f} m/s\n"
        repr += f"Characteristic impedance Z0 = {self.Z0:.0f} Ohm\n"
        repr += f"Equivalent impedance Ceq = {self.Ceq*1e15:.2f} fF\n"
        repr += f"Equivalent inductance Leq = {self.Leq*1e9:.1f} nH\n"
        repr += f"Equivalent impedance Zeq = {self.Zeq:.0f} Ohm\n"
        repr += f"Resonance frequency fr = {self.fr*1e-9:.4f} GHz\n"
        
        return repr

# %%
