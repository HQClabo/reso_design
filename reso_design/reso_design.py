
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

# Functions 
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

def calculate_geometric_capacitance_coplanar(w, gap, H, eps_r=11.9):
    K = K_func(w, gap, H)
    Kp = Kp_func(w, gap, H)
    K1 = K1_func(w, gap, H)
    K1p = K1p_func(w, gap, H)
    Gamma = 1 / (K/Kp + K1/K1p)
    A = 2*Gamma
    B = 1/A
    eps_eff = (1 + eps_r * Kp/K*K1/K1p) / (1 + Kp/K*K1/K1p)

    return 4*eps_eff*eps_0 * B

def calculate_junction_kinetic_inductance(w, l, RA):
    area = w*l
    resistance = RA/area
    Ic0 = delta_0*np.pi/(2*resistance)
    return phi_0/(2*np.pi*Ic0)

# Class definitions

'''
Source for material properties: https://uspas.fnal.gov/materials/15Rutgers/1_SRF_Fundamentals.pdf
'''

class Junction:
    """
    Class describing an Josephson junction Al-AlOx-Al.

    Args:
        d_top: Thickness of the top Al layer.
        d_bottom: Thickness of the bottom Al layer.
        t: Thickness of the tunneling barrier.
        w: Width of the junction.
        l: Length of the junction.
        RA: Resistance times area of the junction.

    """
    def __init__(self, d_top, d_bottom, tox, w, l, RA=550e-12):
        self.d_top = d_top
        self.d_bottom = d_bottom
        self.tox = tox
        self.w = w
        self.l_junction = l
        self.RA = RA

        self.london = 16e-9
        self.pippard = 1600e-9
        self.B_crit_bulk = 10e-3

        self.update()
    
    def update(self):
        self.london_eff_up = self.london * np.sqrt(self.pippard/self.d_top)
        self.london_eff_down = self.london * np.sqrt(self.pippard/self.d_bottom)

        self.B_crit_in_up = self.B_crit_bulk * self.london_eff_up / self.d_top * np.sqrt(24)
        self.B_crit_in_down = self.B_crit_bulk * self.london_eff_down / self.d_bottom * np.sqrt(24)
        self.B_crit_in = min(self.B_crit_in_up, self.B_crit_in_down)

        self.B_crit_out = 1.65*phi_0/(self.w)**2

        flux_A = self.w * (self.tox + min(self.london_eff_up, self.d_top) + min(self.london_eff_down, self.d_bottom))
        self.B_phi0 = phi_0 / flux_A

        self.area = self.w*self.l_junction
        self.R_junction = self.RA/self.area
        self.Ic0 = delta_0*np.pi/(2*self.R_junction)
        self.L_junction = phi_0/(2*np.pi*self.Ic0)
        self.EJ = (phi_0/(2*np.pi))**2 /self.L_junction

        self.C_J = eps_0*eps_r_AlOx*self.w*self.l_junction/self.tox
        self.Ec = e**2/(2*self.C_J)
        self.RJ = self.RA/(self.w * self.l_junction)
        self.alpha0 = h/(4*e**2*self.RJ)
        self.f_plasma = 1/np.sqrt(2*np.pi*self.L_junction*self.C_J)

    def f_of_B_in(self, B_in, f_max=10e9):
        return f_max * np.power(1 - (B_in/self.B_crit_in)**2, 1./4) * np.sqrt(np.abs(np.sinc(B_in / self.B_phi0)))

    def plot_f_of_B_in(self, f_max=10e9):
        """
        Plot the modulation of the frequency as a function of the in-plane magnetic field.

        Args:
            f_max: Frequency at zero field.
        """
        B_range = np.linspace(-self.B_crit_in, self.B_crit_in, 1000)
        fig = plt.figure()
        plt.plot(B_range*1e3, self.f_of_B_in(B_range, f_max)*1e-9)
        plt.xlabel("$B_{\parallel}$ (mT)")
        plt.ylabel("f (GHz)")
        return fig
    
    def f_of_B_out(self, B_out, f_max=10e9):
        return f_max * np.power(1 - (B_out/self.B_crit_out)**2, 1./4)
    
    def plot_f_of_B_out(self, f_max=10e9):
        """
        Plot the resonance frequency as a function of an external out-of-plane magnetic field.
        """
        B_range = np.linspace(-self.B_crit_out, self.B_crit_out, 1000)
        fig = plt.figure()
        plt.plot(B_range*1e3, self.f_of_B_out(B_range, f_max)*1e-9)
        plt.xlabel("$B_{\perp}$ (mT)")
        plt.ylabel("f (GHz)")
        return fig

    def print(self):
        print("**********************************************")
        print("JUNCTION PARAMETERS")
        print("**********************************************")
        print("----------------------------------------------")
        print("Geometrical parameters")
        print("----------------------------------------------")
        print(f"Al thickness top: {self.d_top*1e9:.0f} nm")
        print(f"Al thickness bottom: {self.d_bottom*1e9:.0f} nm")
        print(f"Oxide thickness: {self.tox*1e9:.0f} nm")
        print(f"Junction width: {self.w*1e9:.0f} nm")
        print(f"Junction length l: {self.l_junction*1e9:.0f} nm")
        print(f"Junction area: {self.area*1e12:.3f} um^2")

        print("----------------------------------------------")
        print("Behaviour in magnetic field")
        print("----------------------------------------------")
        print(f"London eff. length top: {self.london_eff_up*1e9:.0f} nm")
        print(f"London eff. length bottom: {self.london_eff_down*1e9:.0f} nm")
        print(f"Out-of-plane Bcrit: {self.B_crit_out*1e3:.1f} mT")
        print(f"In-plane Bcrit top: {self.B_crit_in_up*1e3:.1f} mT")
        print(f"In-plane Bcrit bottom: {self.B_crit_in_down*1e3:.1f} mT")
        print(f"In-plane one flux quantum B field: {self.B_phi0*1e3:.1f} mT")

        print("----------------------------------------------")
        print("Junction physical quantity")
        print("----------------------------------------------")
        print(f"Junction RT resistance = {self.R_junction:.0f} Ohm")
        print(f"Critical current Ic0 = {self.Ic0*1e9:.0f} nA")
        print(f"Junction inductance L_junction = {self.L_junction*1e9:.2f} nH/junction")
        print(f"Junction capacitance C_junction = {self.C_J*1e15:.2f} fF")
        print(f"EJ = {self.EJ} J")
        print(f"Ratio EJ/Ec = {self.EJ/self.Ec:.2f}")
        print(f"Alpha0 = RQ/RJ = {self.alpha0:.2f}")
        print(f"Plasma frequency f_P = {self.f_plasma*1e-9:.0f} GHz")


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

    def print(self):
        print("**********************************************")
        print("SUPERCONDUCTING FILM")
        print("**********************************************")
        print(f"Material: {self.material}")
        print(f"Film thickness: {self.d*1e9:.0f} nm")
        print(f"London eff. length: {self.london_eff*1e9:.0f} nm")
        print(f"In-plane Bcrit: {self.B_crit_in*1e3:.1f} mT")
        print(f"Out-of-plane Bcrit: {self.B_crit_bulk*1e3:.1f} mT")
        return repr
    

class JJArrayDolan(Junction):
    def __init__(self, d_top, d_bottom, tox, w, l_junction, l_spurious, l_unit, gap, H, N, RA, eps_r=11.9, type="lambda_quarter"):
        """
        Class describing a JJ array resonator (so far only lambda/4).

        Args:
            d_top: Thickness of the top Al layer.
            d_bottom: Thickness of the bottom Al layer.
            w: Width of the resonator.
            l_junction: Wength of the junction.
            l_spurious: Length of the spurious junction.
            l_unit: Length of one unit (junction + spurious junction + gaps in between, i.e. bridge + body).
            gap: Gap between resonator core and ground plane.
            H: Substrate thickness.
            l_unit: Length of the junction unit.
            N: Number of junctions.
            RA: Junction resistance at room T times area.
            eps_r: Dielectric constant of the substrate (default is 11.9).
            type: "lambda_quarter" or "lambda_half".
        
        """
        # Initialize the junction parameters
        # An now all the other parameters
        self.gap = gap
        self.H = H
        self.l_junction = l_junction
        self.l_spurious = l_spurious
        self.l_unit = l_unit
        self.N = N
        self.RA = RA
        self.eps_r = eps_r
        assert type == "lambda_quarter" or type == "lambda_half"
        self.type = type

        # Initialize to None
        self.Ctot = None

        super().__init__(d_top, d_bottom, tox, w, l_junction)
        self.update()

    def update(self):
        """
        (Re)calculate all the resonator's parameters. 
        This function gets called during the initialization of the object. 
        In addition, you can use this function after changing one of the object's attributes.

        Returns: None
        
        """
        super().update()
    
        self.length = self.l_unit * self.N

        self.L_spurious = calculate_junction_kinetic_inductance(self.w, self.l_spurious, self.RA)
        self.Ltot = (self.L_junction+self.L_spurious)*self.N
        self.L = self.Ltot/self.length
        self.n_sq = int(self.length/self.w)
        self.Lsq = self.Ltot/self.n_sq

        self.C = calculate_geometric_capacitance_coplanar(self.w, self.gap, self.H, self.eps_r)

        # In case Ctot has been written (for instance because obtained from simulation)
        if self.Ctot != None: 
            self.C = self.Ctot

        self.Z0 = np.sqrt(self.L/self.C)
        self.vph = 1/np.sqrt(self.L*self.C)
        self.alpha = self.Z0 / (h/(2*e)**2)

        if self.type == "lambda_quarter":
            self.fr = self.vph/(4*self.length) 
            self.Leq = 8*self.Ltot/(np.pi**2) 
            self.Ceq = self.C*self.length/2
            self.Zeq = 4/np.pi * self.Z0
        if self.type == "lambda_half":
            self.fr = self.vph/(2*self.length)
            self.Leq = 2*self.Ltot*(np.pi**2) 
            self.Ceq = self.C*self.length/2
            self.Zeq = 2/np.pi * self.Z0

         
        self.fr_eq = 1/(2*np.pi*np.sqrt(self.Leq*self.Ceq))


    def update_C_from_simulation(self, fr_simulation):
        self.Ctot = 1/(16 * self.length**2 * fr_simulation**2 * self.L)
        self.update()
    
    def plot_f_of_B_in(self):
        """
        Plot the resonance frequency as a function of an external in-plane magnetic field.
        """
        return super().plot_f_of_B_in(f_max = self.fr)

    def plot_f_of_B_out(self):
        return super().plot_f_of_B_out(f_max = self.fr)

    def print(self):
        super().print()

        print("\n**********************************************")
        print("JJ ARRAY RESONATOR PARAMETERS")
        print("**********************************************")
        print("----------------------------------------------")
        print("Design parameters")
        print("----------------------------------------------")
        print(f"Number of junctions N = {self.N}")
        print(f"Total length l = {self.length*1e6:.2f} um")
        print(f"Junction length l_junction = {self.l_junction*1e6:.1f} um")
        print(f"Length of spurious junction l_spurious = {self.l_spurious*1e6:.1f} um")
        print(f"Unit length = {self.l_unit*1e6:.1f} um")
        print("----------------------------------------------")
        print("Inductance and capacitance parameters")
        print("----------------------------------------------")
        print(f"Total inductance Ltot = {self.Ltot*1e9:.2f} nH")
        print(f"Number of squares: {self.n_sq}")
        print(f"Inductance per square Lsq = {self.Lsq*1e12:.0f} pH/sq")
        print(f"Inductance per unit length L = {self.L*1e3:.2f} nH/um")
        print(f"Capacitance per unit length C = {self.C*1e9:.2f} fF/um")
        print("----------------------------------------------")
        print("Resonator parameters")
        print("----------------------------------------------")
        print(f"Phase velocity vph = {self.vph*1e-8:.4f} 10^8 m/s")
        print(f"Characteristic impedance Z0 = {self.Z0:.0f} Ohm")
        print(f"Equivalent capacitance Ceq = {self.Ceq*1e15:.2f} fF")
        print(f"Equivalent inductance Leq = {self.Leq*1e9:.1f} nH")
        print(f"Equivalent impedance Zeq = {self.Zeq:.0f} Ohm")
        print(f"Resonance frequency fr = {self.fr*1e-9:.4f} GHz")
        print(f"Eq resonance frequency fr = {self.fr_eq*1e-9:.4f} GHz")
        print(f"Alpha = Z/RQ = {self.alpha}")

        
    
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
        self.tox = t
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

# %%
