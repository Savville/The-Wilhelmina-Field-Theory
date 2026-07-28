import os
import sys
import numpy as np
try:
    import ltspice
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ltspice"])
    import ltspice

def extract_sidebands(filepath, nodes, carrier_f, offset_fs):
    l = ltspice.Ltspice(filepath)
    l.parse()
    
    t = l.get_time()
    dt = t[1] - t[0]
    fs = 1.0 / dt
    N = len(t)
    
    freqs = np.fft.rfftfreq(N, d=dt)
    
    results = {n: [] for n in nodes}
    
    for n in nodes:
        v = l.get_data(n)
        V_fft = np.abs(np.fft.rfft(v)) * (2.0 / N)
        
        # Find carrier index
        carrier_idx = np.argmin(np.abs(freqs - carrier_f))
        carrier_power = 10 * np.log10(V_fft[carrier_idx]**2 / 2.0)
        
        for offset in offset_fs:
            sideband_f = carrier_f + offset
            sideband_idx = np.argmin(np.abs(freqs - sideband_f))
            sideband_power = 10 * np.log10(V_fft[sideband_idx]**2 / 2.0)
            
            # Normalised to carrier
            pn_dbc = sideband_power - carrier_power
            results[n].append((offset, pn_dbc))
            
    return results

def update_validation_script(results_array, results_single):
    # Map nodes to the expected dict keys
    # OSC1_end = V(1), OSC2_centre = V(2), OSC3_end = V(3)
    measured_data_str = f"""
MEASURED_DATA = {{
    'OSC1_end':    {results_array['V(1)']},
    'OSC2_centre': {results_array['V(2)']},
    'OSC3_end':    {results_array['V(3)']},
    'Single_isolated': {results_single['V(1)']},
}}
"""
    
    with open("wilhelmina_validation.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace the MEASURED_DATA block
    start_str = "MEASURED_DATA = {"
    end_str = "}"
    
    start_idx = content.find(start_str)
    # Find the closing brace of MEASURED_DATA
    end_idx = content.find("}", start_idx) + 1
    
    new_content = content[:start_idx] + measured_data_str.strip() + content[end_idx:]
    
    with open("wilhelmina_validation.py", "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Updated wilhelmina_validation.py with measured LTspice data.")

def main():
    f0 = 1.591549e9
    offsets = [1e5, 1e6] # 100kHz and 1MHz that we injected
    
    print("Processing array...")
    res_array = extract_sidebands("wilhelmina_array.raw", ["V(1)", "V(2)", "V(3)"], f0, offsets)
    
    print("Processing single...")
    res_single = extract_sidebands("wilhelmina_single.raw", ["V(1)"], f0, offsets)
    
    update_validation_script(res_array, res_single)

if __name__ == "__main__":
    main()
