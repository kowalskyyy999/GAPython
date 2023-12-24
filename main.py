def main():
    from src.ga import GeneticAlgorithm

    num = [3.019]
    den = [1, 23, 73.75, 22.32]
    n_var = 3                       # Kp, Ki, Kd
    n_bit = 10
    ra = 30                         # Upper bound
    rb = 0                          # lower bound
    population = 100
    minimum_target = 82

    GeneticAlgorithm(
        num,
        den, 
        n_var,
        n_bit,
        ra, 
        rb, 
        population,
        minimum_target
    )()

if __name__ == "__main__":
    main()