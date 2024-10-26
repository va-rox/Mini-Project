import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import erfc

def hammgen(r):
    n = 2**r - 1
    k = n - r
    H = np.array([list(map(int, bin(i)[2:].zfill(r))) for i in range(1, n+1)]).T
    G = np.hstack((np.eye(k), H.T[:k]))
    return H, G

# Parameters
N = 50000  # number of input bits
n1, k1 = 7, 4  # Hamming (7,4)

# Hamming(7,4)
h, g = hammgen(n1 - k1)  # create hamming parity and generator matrix
dictII = np.mod(np.dot(np.arange(2**k1)[:, None] >> np.arange(k1)[::-1] & 1, g), 2)  # create dictionary

# Create random bit_streams messages matrix N streams-messages of size k
msg4 = (np.sign(np.random.randn(N, k1)) + 1) / 2
msg4_m = msg4.copy()  # modulated messages
msg4_m[msg4_m == 0] = -1  # BPSK Mapping

# Create hamming coded messages
coded_w7 = np.mod(np.dot(msg4, g), 2)
coded_w7[coded_w7 == 0] = -1  # modulation one bit per symbol

EbN0dB = np.arange(0, 11)  # SNR vector represented in dB
EbN0 = 10**(EbN0dB / 10)

# BER vectors
ber_noCoding = np.zeros(len(EbN0))
ber_7sdd = np.zeros(len(EbN0))
ber_7hdd = np.zeros(len(EbN0))

for i in range(len(EbN0)):
    yI = (np.sqrt(EbN0[i]) * msg4_m) + np.random.randn(N, k1)
    yII = (np.sqrt(EbN0[i]) * coded_w7) + np.random.randn(N, n1)  # channel out signal with AWGN (Nxn1) matrix
    for j in range(N):
        # No coding case
        yI_hd = yI[j, :]
        yI_hd[yI_hd > 0] = 1
        yI_hd[yI_hd <= 0] = 0
        ber_noCoding[i] += np.sum(yI_hd != msg4[j, :]) / k1

        # Hamming (7,4)
        # Soft Decision Decoding
        distanceII = np.linalg.norm(yII[j, :] - dictII, axis=1)  # Hamming distance
        decoded_w7 = dictII[np.argmin(distanceII), :]
        cw7 = coded_w7[j, :]
        cw7[cw7 == -1] = 0  # revert modulation
        ber_7sdd[i] += np.sum(decoded_w7 != cw7) / n1

        # Hard Decision Decoding
        yII_hd = yII[j, :]
        yII_hd[yII_hd > 0] = 1
        yII_hd[yII_hd <= 0] = 0
        zII = np.dot(h, yII_hd.T) % 2  # calculate syndrome
        for g1 in range(len(yII_hd)):
            if np.array_equal(zII, h[:, g1]):
                yII_hd[g1] = 1 - yII_hd[g1]  # fix mistake using syndrome
        ber_7hdd[i] += np.sum(yII_hd != cw7) / n1

# Final average BER
ber_noCoding /= N
ber_7sdd /= N
ber_7hdd /= N

plt.figure(1)
plt.semilogy(EbN0dB, ber_noCoding, "b--", label="No coding")
plt.semilogy(EbN0dB, ber_7sdd, "r-s", label="Hamming7-SDD")
plt.semilogy(EbN0dB, ber_7hdd, "k-s", label="Hamming7-HDD")
plt.grid(True)
plt.title("BER for no coding and (7,4) Hamming Code")
plt.ylabel("BER")
plt.xlabel("Eb/N0(dB)")
plt.legend()
plt.axis([min(EbN0dB), max(EbN0dB), 1e-4, 1])
plt.show()