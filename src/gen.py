import numpy as np
from src.kromosom import generate_kromosom  

def generate_gen(n_var, n_bit, ra, rb):
    gen = []
    kromosom = generate_kromosom(n_var, n_bit)
    for i in range(1, n_var+1):
        kr = kromosom[n_bit * (i - 1): n_bit * i]
        for j in range(1, n_bit+1):
            kr[j - 1] = kr[j-1] * (2**(n_bit - j))
        x = np.sum(kr)
        x = x/((2**n_bit) -1)
        gen.append(rb + (ra - rb) * x)
    
    return gen, kromosom
