
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0][:-1], [r[:-1] for r in rows[1:]], [r[-1].strip().lower() == "yes" for r in rows[1:]]

def covers(h, x):
    return all(hi == "?" or hi == xi for hi, xi in zip(h, x))

def more_general(h1, h2):  # h1 >= h2
    return all(a == "?" or a == b for a, b in zip(h1, h2))

def candidate_elimination(X, y):
    n = len(X[0])
    domains = [sorted({x[i] for x in X}) for i in range(n)]
    S = [tuple(["0"] * n)]
    G = [tuple(["?"] * n)]
    history = [(len(S), len(G))]
    print(f"Initial\n  S = {S}\n  G = {G}\n")

    for t, (x, label) in enumerate(zip(X, y), 1):
        if label:  # positive example
            G = [g for g in G if covers(g, x)]
            newS = []
            for s in S:
                if covers(s, x):
                    newS.append(s)
                else:
                    h = tuple(xi if si == "0" else (si if si == xi else "?")
                              for si, xi in zip(s, x))
                    if any(more_general(g, h) for g in G):
                        newS.append(h)
            S = [s for s in set(newS)
                 if not any(s != o and more_general(s, o) for o in newS)]
        else:      # negative example
            S = [s for s in S if not covers(s, x)]
            newG = []
            for g in G:
                if not covers(g, x):
                    newG.append(g)
                    continue
                for i in range(n):
                    if g[i] == "?":
                        for v in domains[i]:
                            if v != x[i]:
                                h = g[:i] + (v,) + g[i + 1:]
                                if any(more_general(h, s) for s in S):
                                    newG.append(h)
            G = [g for g in set(newG)
                 if not any(g != o and more_general(o, g) for o in newG)]
        history.append((len(S), len(G)))
        print(f"After example {t} ({'+' if label else '-'}) {x}\n  S = {S}\n  G = {G}\n")
    return S, G, history

def plot(attrs, S, G, history):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    steps = range(len(history))
    a1.plot(steps, [h[0] for h in history], "o-", label="|S| (specific boundary)")
    a1.plot(steps, [h[1] for h in history], "s--", label="|G| (general boundary)")
    a1.set_xlabel("Training examples processed"); a1.set_ylabel("Hypotheses in boundary")
    a1.set_xticks(list(steps)); a1.set_yticks(range(0, max(max(h) for h in history) + 2))
    a1.set_title("Boundary sizes during Candidate-Elimination"); a1.legend(); a1.grid(alpha=.3)

    a2.axis("off"); a2.set_title("Final version space boundaries")
    rows = [["S"] + list(s) for s in S] + [["G"] + list(g) for g in G]
    tbl = a2.table(cellText=rows, colLabels=["Set"] + attrs, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    plt.tight_layout(); plt.savefig("final_graph.png", dpi=150)

if __name__ == "__main__":
    attrs, X, y = load("training_data.csv")
    S, G, hist = candidate_elimination(X, y)
    print("FINAL SPECIFIC BOUNDARY S:", S)
    print("FINAL GENERAL  BOUNDARY G:", G)
    plot(attrs, S, G, hist)
