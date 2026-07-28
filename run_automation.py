import os
import subprocess
import numpy as np
import scipy.signal as signal

def create_netlist():
    netlist = """* Wilhelmina 3-Node Oscillator Array Validation
* Osc 1
C1 1 0 1p
L1 1 0 10n
R1 1 0 500
G1 1 0 value={0.004*V(1) - 0.002*V(1)**3}

* Osc 2
C2 2 0 1p
L2 2 0 10n
R2 2 0 500
G2 2 0 value={0.004*V(2) - 0.002*V(2)**3}

* Osc 3
C3 3 0 1p
L3 3 0 10n
R3 3 0 500
G3 3 0 value={0.004*V(3) - 0.002*V(3)**3}

* Coupling (K/gamma = 1.0 => R_c = 500 ohms)
Rc1 1 2 500
Rc2 2 3 500

* Inject single-tone "noise" at offset frequencies to measure transfer function
* 1.591549 GHz is f0. We inject tones at f0 + 100kHz and f0 + 1MHz
I_inj 1 0 SINE(0 1e-4 1.591649G)
I_inj2 1 0 SINE(0 1e-4 1.592549G)

.tran 0 2u 1u 1p
.options plotwinsize=0
.save V(1) V(2) V(3)
.end
"""
    with open("wilhelmina_array.net", "w") as f:
        f.write(netlist)
        
    netlist_single = """* Single Isolated Oscillator
C1 1 0 1p
L1 1 0 10n
R1 1 0 500
G1 1 0 value={0.004*V(1) - 0.002*V(1)**3}

I_inj 1 0 SINE(0 1e-4 1.591649G) 
I_inj2 1 0 SINE(0 1e-4 1.592549G) 

.tran 0 2u 1u 1p
.options plotwinsize=0
.save V(1)
.end
"""
    with open("wilhelmina_single.net", "w") as f:
        f.write(netlist_single)

def run_ltspice():
    ltspice_path = r"C:\Users\User\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
    print("Running array simulation...")
    subprocess.run([ltspice_path, "-b", "-Run", "wilhelmina_array.net"])
    print("Running single simulation...")
    subprocess.run([ltspice_path, "-b", "-Run", "wilhelmina_single.net"])

def main():
    create_netlist()
    run_ltspice()
    print("Simulations complete. Ready for analysis.")

if __name__ == "__main__":
    main()
