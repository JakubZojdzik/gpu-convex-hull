import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt


f = open("results.txt", "r").read()

sections = f.split("\n\n")

@dataclass
class Stats:
    alg_name: str
    hull_size: list[int]
    first_run: list[float]
    all_runs: list[float]


N = 0
paper = Stats("GPU QuickHull Paper", [], [], [])
naive = Stats("GPU QuickHull Naive", [], [], [])
cpu = Stats("CPU Monotone Chain", [], [], [])

objs = [paper, naive, cpu]
res_N = []
res_alg = []
res_avg_first = []
res_avg_all = []
res_avg_resp = []

for s in sections:
    if "N = " in s:
        if N != 0:
            for obj in objs:
                res_N.append(N / 10**6)
                res_alg.append(obj.alg_name)
                res_avg_first.append(round(sum(obj.first_run) / len(obj.first_run), 4))
                res_avg_all.append(round(sum(obj.all_runs) / len(obj.all_runs), 4))
                res_avg_resp.append(round(sum(obj.hull_size) / len(obj.hull_size), 4))
                obj.hull_size = []
                obj.first_run = []
                obj.all_runs = []

        N = int(s.split("\n")[1].split()[-1])
        continue
    
    lines = s.splitlines()
    cpu.hull_size.append(int(lines[2].split()[-1]))
    cpu.first_run.append(float(lines[3].split()[-2]))
    cpu.all_runs += [float(x) for x in lines[4].split()[3:] if x != "ms"]

    paper.hull_size.append(int(lines[6].split()[-1]))
    paper.first_run.append(float(lines[7].split()[-2]))
    paper.all_runs += [float(x) for x in lines[8].split()[3:] if x != "ms"]
    
    naive.hull_size.append(int(lines[10].split()[-1]))
    naive.first_run.append(float(lines[11].split()[-2]))
    naive.all_runs += [float(x) for x in lines[12].split()[3:] if x != "ms"]

df = pd.DataFrame({
    "N": res_N,
    "Algorithm": res_alg,
    "Avg First Run (ms)": res_avg_first,
    "Avg All Runs (ms)": res_avg_all,
    "Avg Hull Size": res_avg_resp,
})

print(df)

plt.figure()
for alg, g in df.groupby("Algorithm"):
    plt.plot(g["N"], g["Avg All Runs (ms)"], marker='o', label=alg)

plt.xlabel("N (milions)")
plt.ylabel("time")
plt.title("Results")
plt.legend()
plt.grid(True, which="both")
plt.show()
