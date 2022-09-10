import numpy as np
import torch

_PAD = 0
_GO = 1
_EOS = 2
TOKENS = [_PAD,_GO,_EOS]
# mass
mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {
    '_PAD': 0.0,
    '_GO': mass_N_terminus - mass_H,
    '_EOS': mass_C_terminus + mass_H,
    'A': 71.03711,  # 0
    'R': 156.10111,  # 1
    'N': 114.04293,  # 2
    'N(+.98)': 115.02695,
    'D': 115.02694,  # 3
    #~ 'C': 103.00919, # 4
    'C(+57.02)': 160.03065,  # IAA fixed modification
    #~ 'Cmod': 161.01919, # C(+58.01)
    'E': 129.04259,  # 5
    'Q': 128.05858,  # 6
    'Q(+.98)': 129.0426,  # ?
    'G': 57.02146,  # 7
    'H': 137.05891,  # 8
    'I': 113.08406,  # 9
    'L': 113.08406,  # 10
    'K': 128.09496,  # 11
    'M': 131.04049,  # 12
    'M(+15.99)': 147.0354,  # sulfoxide
    'F': 147.06841,  # 13
    'P': 97.05276,  # 14
    'S': 87.03203,  # 15
    'T': 101.04768,  # 16
    'W': 186.07931,  # 17
    'Y': 163.06333,  # 18
    'V': 99.06841,  # 19
}
vocab = dict([(x, y) for (y, x) in enumerate(mass_AA.keys())])  #str to int
vocab_size = len(mass_AA)
masses = dict(enumerate(mass_AA.values()))  #int to mass
masses_np = np.array(list(mass_AA.values()), np.float64)
mask = [1 if i not in TOKENS else 0 for i in range(vocab_size)]
# preprocessing
CHARGE = 1
MAX_MZ = 3000.0
MAX_LEN = 30
MS_RESOLUTION = 10.0
MZ_SIZE = int(MAX_MZ * MS_RESOLUTION)
_buckets = [12, 22, 32]
num_ion = 8

# training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
BATCH_SIZE = 128
TEACHER_FORCING_RATIO = 0.5
# dropout probility
P_CONV = 0.25
P_DENSE = 0.5

# Ion_CNN
VOCAB_SIZE = 26
FRAGMENT_ION = 8
WINDOW_SIZE = 10
kernel = 64
ION_SIZE = kernel * FRAGMENT_ION * (WINDOW_SIZE//2)
MAX_GRADIENT_NORM = 10
# LSTM
INPUT_SIZE = 512
HIDDEN_SIZE = 512

vocab_reverse = [
    'PAD',
    'GO',
    'EOS',
    'A',
    'R',
    'N',
    'N*',
    'D',
    #~ 'C',
    'C*',
    'E',
    'Q',
    'Q*',
    'G',
    'H',
    'I',
    'L',
    'K',
    'M',
    'M*',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'V',
]
