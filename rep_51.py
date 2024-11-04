import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Parameters
N = 50000  # Length of data bit stream
m = np.random.randint(0, 2, N)  # Random 0s and 1s
c = np.repeat(m, 5)  # Repetition Coding (5,1)

# BPSK Mapping
x = np.where(c == 0, -1, 1)

r = 1/5  # Code rate
rep = 5  # Block size
BER_sim = []
BER_th_C = []
BER_UC = []
BER_sim_SDD = []

for EbN0dB in range(11):
    EbN0 = 10**(EbN0dB/10)
    sigma = np.sqrt(1/(2*r*EbN0))  # Noise Standard deviation
    n = sigma * np.random.randn(len(x))  # Random Noise (AWGN) with adjusted variance
    y = x + n  # Received symbol = Transmitted + Noise
    c_cap = (y > 0).astype(int)  # If y is positive, c_cap=1, otherwise c_cap=0

    m_cap = []
    m_cap_SDD = []

    for j in range(len(c_cap) // rep):
        code = c_cap[j*rep:(j+1)*rep]  # Storing one block of symbols in single variable (5 bits)
        code1 = 1 if np.sum(code) >= 3 else 0
        m_cap.append(code1)

        code_SDD = y[j*rep:(j+1)*rep]
        code2 = 1 if np.sum(code_SDD) > 0 else 0
        m_cap_SDD.append(code2)

    m_cap = np.array(m_cap)
    m_cap_SDD = np.array(m_cap_SDD)

    noe = np.sum(m != m_cap)  # Number of Errors HDD
    ber_sim1 = noe / N
    BER_sim.append(ber_sim1)  # Appending the BER values in an array

    noe1 = np.sum(m != m_cap_SDD)  # Number of Errors SDD
    ber_sim2 = noe1 / N
    BER_sim_SDD.append(ber_sim2)  # Appending the BER values in an array

    p = 0.5 * erfc(np.sqrt(r * EbN0))  # Single bit Probability of Error in Coded BPSK
    ber_th_q = 5 * (p**4) * (1 - p) + p**5 + 10 * (1 - p**2) * (p**3)  # 3-bit Error + 4-bit Error
    BER_th_C.append(ber_th_q)  # Theoretical BER for Coded BPSK
    BER_UC.append(0.5 * erfc(np.sqrt(EbN0)))  # BER for Uncoded BPSK

EbN0dB = np.arange(11)
plt.figure(1)
plt.semilogy(EbN0dB, BER_sim, 'b*-', label="Simulated HDD(5,1)")
plt.semilogy(EbN0dB, BER_sim_SDD, 'gp-', label="Simulated SDD(5,1)")
plt.semilogy(EbN0dB, BER_UC, 'k--', label="Uncoded")
plt.title("BER Analysis of (5,1) Repetition Codes")
plt.xlabel('Eb/N0(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.legend()
plt.axis([min(EbN0dB), max(EbN0dB), 1e-4, 1])
plt.show()