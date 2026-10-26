"""Regenerate fig5a-d on the CANONICAL n=7,435 subset with correct text-based
doc_id->metadata alignment.
"""
import json, os, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
OUT  = Path(os.environ.get("SAUDADE_FIG_DIR", BASE / "figures"))
OUT.mkdir(parents=True, exist_ok=True)

pkg = json.load(open(BASE/"canonical_full_package.json", encoding="utf-8"))
M = pkg["models"]

# ---- style config ----
STYLE = {
 "AMALIA-9B-DPO": {"family":"Fully Open","color":"#882255","marker":"p"},
 "AMALIA-9B-SFT": {"family":"Fully Open","color":"#AA3377","marker":"s"},
 "EuroLLM-9B":    {"family":"Fully Open","color":"#CCBB44","marker":"v"},
 "Gervasio-8B":   {"family":"Fully Open","color":"#EE6677","marker":"<"},
 "OLMo 2-7B":     {"family":"Fully Open","color":"#4477AA","marker":"o"},
 "Apertus-8B":    {"family":"Fully Open","color":"#228833","marker":"^"},
 "Salamandra-7B": {"family":"Fully Open","color":"#66CCEE","marker":"d"},
 "Gemma 3-12B":   {"family":"Open-Weight","color":"#CC6677","marker":"p"},
 "Gemma 2-9B":    {"family":"Open-Weight","color":"#DDCC77","marker":"s"},
 "Qwen 3-8B":     {"family":"Open-Weight","color":"#999933","marker":"<"},
 "Ministral-8B":  {"family":"Open-Weight","color":"#88CCEE","marker":"d"},
 "Qwen 2.5-7B":   {"family":"Open-Weight","color":"#117733","marker":"v"},
 "Mistral-7B":    {"family":"Open-Weight","color":"#332288","marker":"o"},
 "LLaMA 3.1-8B":  {"family":"Open-Weight","color":"#44AA99","marker":"^"},
}
DT_BINS   = ["1-2","3-5","6-10","11-30",">30"]
DT_LABELS = ["1\u20132 y\n(Fine)","3\u20135 y","6\u201310 y","11\u201330 y",">30 y\n(Coarse)"]
ERA_BINS   = ["<1900","1900-1950","1950-1974","1974-2000","2000+"]
ERA_LABELS = ["<1900\n(Ancient)","1900\u201350\n(I Rep.)","1950\u201374\n(Est. Novo)","1974\u201300\n(Dem./EEC)","2000+\n(Contemp.)"]

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    margin = z*np.sqrt((p*(1-p) + z*z/(4*n))/n)/denom
    return (max(0.0,(center-margin)*100), min(100.0,(center+margin)*100))

plt.rcParams.update({
    'font.family':'sans-serif','font.size':11,'axes.labelsize':12,
    'axes.titlesize':13,'xtick.labelsize':9.5,'ytick.labelsize':10,
    'legend.fontsize':8.5,'figure.titlesize':14,'figure.dpi':300,
    'lines.linewidth':1.8,'lines.markersize':6,'grid.alpha':0.4,'grid.linestyle':'--'})

def best(m): return M[m]["best_mode"]

# ============================================================
# Figure 5a: |Δt| resolution curves, 2 panels
# ============================================================
def curve_panel(ax, models, xbins, xlabels, key_acc, key_n, title):
    ax.set_facecolor("#FAFAFA"); ax.grid(True, zorder=0)
    sm = sorted(models, key=lambda m: -M[m][best(m)]["acc"])
    for m in sm:
        c = M[m][best(m)]
        st = STYLE[m]
        accs = np.array([c[key_acc][b] for b in xbins])
        ns   = np.array([c[key_n][b] for b in xbins])
        ks   = np.round(accs*ns/100.0)
        lo = np.array([wilson_ci(k,n)[0] for k,n in zip(ks,ns)])
        hi = np.array([wilson_ci(k,n)[1] for k,n in zip(ks,ns)])
        x = np.arange(len(xbins))
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, zorder=2)
        ax.plot(x, accs, label=f"{m} ({c['acc']:.1f}%)", color=st["color"],
                marker=st["marker"], zorder=3)
    ax.set_xticks(range(len(xbins))); ax.set_xticklabels(xlabels)
    ax.axhline(50.0, color="gray", linestyle=":", linewidth=1.2, zorder=1)
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.legend(loc="upper left", frameon=True, edgecolor="#CCCCCC", fontsize=8.2)

fo = [m for m in STYLE if STYLE[m]["family"]=="Fully Open"]
ow = [m for m in STYLE if STYLE[m]["family"]=="Open-Weight"]

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
curve_panel(axes[0], fo, DT_BINS, DT_LABELS, "dt_acc","dt_n","Fully Open Models")
curve_panel(axes[1], ow, DT_BINS, DT_LABELS, "dt_acc","dt_n","Open-Weight Models")
axes[0].set_xlabel("Temporal Distance Resolution ($|\\Delta t|$)")
axes[1].set_xlabel("Temporal Distance Resolution ($|\\Delta t|$)")
for ax in axes: ax.set_ylim(38, 96)
fig.suptitle("Empirical Accuracy vs. Temporal Distance ($|\\Delta t|$) on SAUDADE  (canonical $n=7{,}435$)",
             fontweight="bold", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT/"fig5a_deltat.pdf", bbox_inches="tight")
fig.savefig(OUT/"fig5a_deltat.png", bbox_inches="tight", dpi=200)
plt.close(fig); print("saved fig5a")

# ============================================================
# Figure 5b: historical era curves, 2 panels
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
curve_panel(axes[0], fo, ERA_BINS, ERA_LABELS, "era_acc","era_n","Fully Open Models")
curve_panel(axes[1], ow, ERA_BINS, ERA_LABELS, "era_acc","era_n","Open-Weight Models")
axes[0].set_xlabel("Historical Epoch (Chronological Anchor)")
axes[1].set_xlabel("Historical Epoch (Chronological Anchor)")
for ax in axes: ax.set_ylim(38, 96)
fig.suptitle("Empirical Accuracy across Historical Eras on SAUDADE  (canonical $n=7{,}435$)",
             fontweight="bold", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT/"fig5b_era.pdf", bbox_inches="tight")
fig.savefig(OUT/"fig5b_era.png", bbox_inches="tight", dpi=200)
plt.close(fig); print("saved fig5b")

# ============================================================
# Figure 5c: FRP vs ED diagnostic quadrant (CLEAN, NO OVERLAP)
# ============================================================
frps = {m: M[m][best(m)]["frp"] for m in STYLE}
eds  = {m: M[m][best(m)]["ed"]  for m in STYLE}
r = np.corrcoef(list(frps.values()), list(eds.values()))[0,1]

fig, ax = plt.subplots(figsize=(10.5, 7.2))
ax.set_facecolor("#FFFFFF")
TAU = 2.0
xmin, xmax = -2.0, 30.0
ymin, ymax = -1.6, 5.0

# quadrant tints
ax.fill_between([xmin,TAU],[TAU,TAU],ymax, color="#EBF3FB", alpha=0.45, zorder=0)   # Coverage-Limited
ax.fill_between([TAU,xmax],[TAU,TAU],ymax, color="#FDEEE9", alpha=0.45, zorder=0)   # Doubly-Limited
ax.fill_between([xmin,TAU],[ymin,ymin],TAU, color="#EAF7EC", alpha=0.45, zorder=0)  # Robust
ax.fill_between([TAU,xmax],[ymin,ymin],TAU, color="#FEF8E7", alpha=0.45, zorder=0)  # Resolution-Limited

ax.axvline(TAU, color="#555", ls="--", lw=1.2, zorder=2)
ax.axhline(TAU, color="#555", ls="--", lw=1.2, zorder=2)
ax.axvline(0, color="#C0C0C0", ls=":", lw=0.9, zorder=1)
ax.axhline(0, color="#C0C0C0", ls=":", lw=0.9, zorder=1)

ax.text(xmin+0.5, ymax-0.3, "COVERAGE-LIMITED\n(ED > 2.0, FRP \u2264 2.0)", fontsize=9, color="#1D4E89", fontweight="bold", va="top", ha="left")
ax.text(xmax-0.5, ymax-0.3, "DOUBLY-LIMITED\n(ED > 2.0, FRP > 2.0)", fontsize=9, color="#8B261D", fontweight="bold", va="top", ha="right")
ax.text(xmin+0.5, ymin+0.25, "ROBUST\n(ED \u2264 2.0, FRP \u2264 2.0)", fontsize=9, color="#1E6527", fontweight="bold", va="bottom", ha="left")
ax.text(xmax-0.5, ymin+0.25, "RESOLUTION-LIMITED\n(ED \u2264 2.0, FRP > 2.0)", fontsize=9, color="#8C6218", fontweight="bold", va="bottom", ha="right")

pos = {
    "Salamandra-7B": (1.5, 0.45, "left", "bottom"),
    "Apertus-8B":    (12.5, 0.85, "right", "bottom"),
    "EuroLLM-9B":    (13.5, -0.65, "right", "top"),
    "Mistral-7B":    (15.7, -0.55, "center", "top"),
    "Gervasio-8B":   (18.7, 0.45, "center", "bottom"),
    "LLaMA 3.1-8B":  (15.0, 1.85, "right", "bottom"),
    "OLMo 2-7B":     (19.2, 2.15, "left", "bottom"),
    "Ministral-8B":  (20.5, -0.75, "right", "top"),
    "Qwen 3-8B":     (24.5, -0.55, "left", "top"),
    "Gemma 3-12B":   (22.8, 1.65, "center", "bottom"),
    "AMALIA-9B-DPO": (25.0, 1.05, "left", "center"),
    "Gemma 2-9B":    (25.0, 0.35, "left", "center"),
    "AMALIA-9B-SFT": (21.5, 4.10, "left", "bottom"),
    "Qwen 2.5-7B":   (26.5, 2.45, "left", "bottom"),
}

for m in STYLE:
    x, y = frps[m], eds[m]
    st = STYLE[m]
    ax.scatter(x, y, color=st["color"], marker=st["marker"], s=160, edgecolors="#111", linewidth=1.3, zorder=6)
    tx, ty, ha, va = pos[m]
    ax.annotate(
        f"{m} ({M[m][best(m)]['acc']:.1f}%)",
        xy=(x, y), xytext=(tx, ty),
        fontsize=8.5, fontweight="bold", color="#151515", ha=ha, va=va, zorder=7,
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.9, shrinkA=3, shrinkB=4)
    )

ax.set_xlabel("Fine-Resolution Penalty (FRP, pp)\n$\\mathrm{Acc}(>30\\,\\mathrm{y})-\\mathrm{Acc}(1\\text{--}2\\,\\mathrm{y})$  [higher = worse fine discrimination]", fontsize=11, labelpad=8)
ax.set_ylabel("Historical Era Decay (ED, pp)\n$\\max(\\mathrm{Modern})-\\mathrm{Acc}(<1900)$  [higher = worse ancient amnesia]", fontsize=11, labelpad=8)
ax.set_title(f"Temporal Error Quadrant (Pearson $r = {r:+.2f}$) \u2014 canonical $n=7{{,}}435$", fontsize=12.5, fontweight="bold", pad=12)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.grid(True, ls=":", alpha=0.5, color="#BBB", zorder=1)

plt.tight_layout()
fig.savefig(OUT/"fig5c_quadrant.pdf", bbox_inches="tight")
fig.savefig(OUT/"fig5c_quadrant.png", bbox_inches="tight", dpi=300)
plt.close(fig); print("saved fig5c")

# ============================================================
# Figure 5d: FRP forest plot with bootstrap CIs
# ============================================================
rows = []
for m in STYLE:
    c = M[m][best(m)]
    lo,hi = M[m]["frp_boot_ci"]
    rows.append({"model":m,"fam":STYLE[m]["family"],"frp":c["frp"],"lo":lo,"hi":hi,
                 "sig":M[m]["frp_excludes_zero"]})
rows.sort(key=lambda r:r["frp"])
fig, ax = plt.subplots(figsize=(8.2, 6.2))
ax.axvspan(-TAU,TAU,color="0.92",zorder=0)
ax.axvline(0,color="0.4",lw=0.8,zorder=1)
fam_col={"Fully Open":"#0072B2","Open-Weight":"#E69F00"}
for i,rr in enumerate(rows):
    c=fam_col[rr["fam"]]
    ax.plot([rr["lo"],rr["hi"]],[i,i],color=c,lw=1.6,zorder=2,alpha=1.0 if rr["sig"] else 0.55)
    if rr["sig"]:
        ax.scatter([rr["frp"]],[i],s=42,color=c,zorder=3,edgecolors="black",linewidths=0.6)
    else:
        ax.scatter([rr["frp"]],[i],s=42,facecolors="white",edgecolors=c,linewidths=1.6,zorder=3)
    ax.text(rr["hi"]+0.7, i, f"{rr['frp']:+.1f} [{rr['lo']:+.1f}, {rr['hi']:+.1f}]",
            va="center",ha="left",fontsize=7.8,color="0.25")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["model"] for r in rows],fontsize=8.5)
ax.set_xlabel("Fine-Resolution Penalty, FRP = acc($>$30 y) $-$ acc(1\u20132 y)  [pp]",fontsize=9.5,labelpad=6)
ax.set_xlim(-6, max(r["hi"] for r in rows)+7.5)
ax.set_ylim(-0.8, len(rows)+0.6)
ax.text(TAU, len(rows)+0.15, "practical equivalence zone ($\\tau = \\pm 2.0$ pp)",
        ha="center",va="bottom",fontsize=8.0,color="0.35",style="italic")
handles=[Line2D([0],[0],color=fam_col["Fully Open"],lw=2,label="Fully Open"),
         Line2D([0],[0],color=fam_col["Open-Weight"],lw=2,label="Open-Weight"),
         Line2D([0],[0],marker="o",color="none",markerfacecolor="white",
                markeredgecolor="0.3",markersize=7,label="CI includes 0 (n.s.)"),
         Line2D([0],[0],marker="o",color="none",markerfacecolor="0.3",
                markersize=7,label="CI excludes 0")]
ax.legend(handles=handles,loc="center left", bbox_to_anchor=(0.04, 0.76),fontsize=8.0,framealpha=0.92,edgecolor="#CCCCCC")
ax.set_title("FRP per model, 95% bootstrap CI (10,000 resamples) \u2014 canonical $n=7{,}435$",fontsize=10.5,fontweight="bold",pad=16)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT/"fig5d_bootstrap_frp.pdf", bbox_inches="tight")
fig.savefig(OUT/"fig5d_bootstrap_frp.png", bbox_inches="tight", dpi=200)
plt.close(fig); print("saved fig5d")

print("\nAll 4 figures saved successfully in:", OUT)
