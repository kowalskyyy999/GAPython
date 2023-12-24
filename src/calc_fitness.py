import numpy as np
from control import TransferFunction
import control

def calculation(num, den, gen):
    tf_sys = TransferFunction(num, den)

    kp, ki, kd = gen[0], gen[1], gen[2]

    tf_pid = TransferFunction(np.array([kp, ki, kd]), np.array([1, 0]))

    tf_mul = tf_sys * tf_pid
    tf_sys_pid = tf_mul.feedback()

    try:
        result = control.step_info(tf_sys_pid)

        ess = result["SteadyStateValue"]
        rise_time = result["RiseTime"]
        settling_time = result["SettlingTime"]
        overshoot = result["Overshoot"]

        fitness_1 = 1/(rise_time + 1) * 100
        fitness_2 = 1/(ess + 0.1) * 100

        if ess == 0:
            fitness_3 = 100
        else:
            fitness_3 = 1/(overshoot + 1) * 100

        if settling_time <= 10:
            fitness_4 = 100
        else:
            fitness_4 = 1/(settling_time+0.01) * 100

        fitness = (fitness_1 + fitness_2 + fitness_3 + fitness_4) / 4

        return fitness
    
    except BaseException as e:
        print(f"Kp: {kp} - Ki: {ki} - Kd: {kd}")
        print(tf_sys_pid)
        print(e)
        return 0        

def calculation_mutation(mutant, num, den,  n_var, n_bit, ra, rb):
    gen = []
    kromosom = mutant["kromosom"]
    for i in range(1, n_var+1):
        kr = kromosom[n_bit * (i - 1): n_bit * i]
        for j in range(1, n_bit+1):
            kr[j - 1] = kr[j-1] * (2**(n_bit - j))
        x = np.sum(kr)
        x = x/((2**n_bit) -1)
        gen.append(rb + (ra - rb) * x)

    fitness = calculation(num, den, gen)

    return fitness, gen
    