import numpy as np 
from random import random

def generate_kromosom(n_var, n_bit):
    jumlah_gen = n_var * n_bit
    kromosom = []
    for _ in range(jumlah_gen):
        if random() > 0.5:
            kromosom.append(1)
        else:
            kromosom.append(0)
    return kromosom
