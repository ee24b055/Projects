import numpy as np

def get_mzi_s(delta_phi_deg):
    """Calculates the 4x4 S-matrix for a bidirectional MZI given phase in DEGREES."""
    delta_phi_rad = np.radians(delta_phi_deg)
    t = 1j * np.sin(delta_phi_rad / 2.0)
    r = np.cos(delta_phi_rad / 2.0)
    return np.array([[0, 0, t, r], 
                     [0, 0, r, t], 
                     [t, r, 0, 0], 
                     [r, t, 0, 0]], dtype=complex)

def solve_mesh(phase_errors_deg, input_port='I1'):
    num_mzis = 14
    total_ports = num_mzis * 4
    
    # Square mesh topology coordinate routing
    connections = [
        ((1,2), (4,0)), ((1,3), (3,0)), ((2,2), (3,1)), ((2,3), (5,1)),
        ((4,2), (7,0)), ((4,3), (6,0)), ((3,2), (6,1)), ((3,3), (9,1)),
        ((5,2), (9,3)), ((5,3), (8,1)), ((7,2), (10,0)), ((7,3), (9,0)),
        ((6,2), (9,2)), ((6,3), (8,0)), ((10,2), (13,0)), ((10,3), (12,0)),
        ((8,2), (12,1)), ((8,3), (11,1)), ((12,2), (13,1)), ((12,3), (14,0)),
        ((11,2), (14,1))
    ]
    
    adj = {}
    for (m1, p1), (m2, p2) in connections:
        idx1, idx2 = (m1 - 1) * 4 + p1, (m2 - 1) * 4 + p2
        adj[idx1], adj[idx2] = idx2, idx1

    external_ports = {
        'I1': (1,0), 'I2': (1,1), 'I3': (2,0), 'I4': (2,1),
        'O1': (13,2), 'O2': (13,3), 'O3': (14,2), 'O4': (14,3)
    }
    
    M = np.eye(total_ports, dtype=complex)
    Source = np.zeros(total_ports, dtype=complex)
    
    for m in range(1, num_mzis + 1):
        S_local = get_mzi_s(phase_errors_deg[m-1])
        base = (m - 1) * 4
        for p_out in range(4):
            g_out = base + p_out
            if g_out in adj:
                g_in = adj[g_out]
                for p_in in range(4):
                    M[g_in, base + p_in] -= S_local[p_out, p_in]

    in_mzi, in_p = external_ports[input_port]
    Source[(in_mzi - 1) * 4 + in_p] = 1.0 
    
    V_in = np.linalg.solve(M, Source)
    
    powers = {}
    for name, (m, p) in external_ports.items():
        base = (m - 1) * 4
        S_local = get_mzi_s(phase_errors_deg[m-1])
        out_amp = sum(S_local[p, pi] * V_in[base + pi] for pi in range(4))
        powers[name] = np.abs(out_amp)**2 * 100
        
    return powers

if __name__ == "__main__":
    # Prompt the user for direct input in degrees
    print("Enter the 14 MZI phase differences in degrees (separated by spaces):")
    user_input = input("> ")
    
    # Parse input numbers
    try:
        cleaned_input = user_input.replace(',', ' ')
        phase_errors_deg = [float(x) for x in cleaned_input.split()]
        if len(phase_errors_deg) != 14:
            raise ValueError
    except ValueError:
        print("\nError: Please enter exactly 14 numerical angle values.")
        exit()
        
    # Solve network matrices
    results = solve_mesh(phase_errors_deg, input_port='I1')
    
    # Simple, clear plain text display with plenty of whitespace
    print("\n\n")
    print("Port      Power Output")
    print("----------------------")
    for port, power in results.items():
        print(f"{port:<8}  {power:6.2f}%")
    print("----------------------")
    print(f"Total     {sum(results.values()):6.2f}%")
    print("\n\n")