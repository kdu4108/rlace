import subprocess
num_classes = list(range(2, 11))
seeds = [1, 2, 3, 4, 5]
for seed in seeds:
    for c in num_classes:
        for rank in range(1, c + 1):
            # print(c, rank, seed)
            subprocess.check_call(["sbatch", "submit_rlace.cluster", f"{c}", f"{rank}", f"{seed}"])
