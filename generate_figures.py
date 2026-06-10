"""
generate_figures.py
===================
Produces all six thesis figures for:

  "Adaptive Concept Drift Detection in Data Streams
   Using Incremental Probabilistic Neural Networks"
  Ahmed Azab — Częstochowa University of Technology, 2026

Figures produced
----------------
  Figure1_Architecture.png     – IPNN framework block diagram
  Figure2_Concept.png          – Conceptual PDF shift + ISE signal illustration
  Figure3_ISE_Timeline.png     – ISE monitoring timeline (real experiment data)
  Figure4_Detection_Results.png – Per-drift delays + Q-parameter sensitivity
  Figure5_PDF_Evolution.png    – IPNN density snapshots from experiment
  Figure6_Stream_Overview.png  – Raw stream + drift markers + rolling statistics

Requirements
------------
  pip install numpy matplotlib

Usage
-----
  # Point at any experiment subfolder:
  python generate_figures.py \
      --exp_dir "experiments/20260415_160256_N20000_..." \
      --out_dir "figures"

  All data is read dynamically from the CSVs inside exp_dir,
  so the script works with any experiment (8-drift, 11-drift, etc.)
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Default paths  (relative to this script)
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_EXP = os.path.join(
    _SCRIPT_DIR, 'experiments',
    '20260415_160256_N20000_drifts11_mean_variance_gradual_variance_'
    'mean_cyclic_distribution_gradual_mean_variance_variance'
)
DEFAULT_OUT = os.path.join(_SCRIPT_DIR, 'figures')

# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────
TYPE_COLORS = {
    'mean':         '#1a5276',
    'variance':     '#c0392b',
    'gradual':      '#1e8449',
    'cyclic':       '#e67e22',
    'distribution': '#7d3c98',
}

THRESHOLD = 0.08
WARMUP    = 300

# Q-sensitivity — multi-experiment comparison, kept as constants
Q_SENSITIVITY = [
    # (Q,   k,   detected, display_label)
    (0.0,   20,   2,  'Q=0\nk=20'),
    (0.0,   30,   2,  'Q=0\nk=30'),
    (0.0,  129,   2,  'Q=0\nk=129'),
    (0.1,   20,   4,  'Q=0.1\nk=20'),
    (0.1,  129,   6,  'Q=0.1\nk=129'),
    (0.2,   20,   6,  'Q=0.2\nk=20'),
    (0.25,   2,   6,  'Q=0.25\nk=2'),
    (0.4,    3,   9,  'Q=0.4\nk=3'),
    (0.5,    3,  11,  'Q=0.5\nk=3\n(optimal)'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def gauss(x, mu, sigma):
    """Pure-NumPy Gaussian PDF — no scipy needed."""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _safe_float(s):
    """Return float or None for empty/nan/None strings."""
    if s in ('', 'None', 'nan', None):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_experiment(exp_dir):
    """
    Read all CSV files from an experiment folder and return a single dict.
    Works with any experiment regardless of number of drifts.
    """
    d = {}

    # ── ISE monitoring ────────────────────────────────────────────────────────
    ise_rows = load_csv(os.path.join(exp_dir, 'csv_03_ise_score.csv'))
    d['ise']         = np.array([float(r['ise_score'])     for r in ise_rows])
    d['alarm_fired'] = np.array([int(r['alarm_fired'])     for r in ise_rows])
    d['all_alarms']  = [int(r['index']) for r in ise_rows if r['alarm_fired'] == '1']

    # ── Ground-truth drifts ───────────────────────────────────────────────────
    gt_rows = load_csv(os.path.join(exp_dir, 'csv_02_drift_ground_truth.csv'))
    d['drift_positions'] = [int(r['position'])       for r in gt_rows]
    d['drift_types']     = [r['type']                for r in gt_rows]
    d['drift_labels']    = [r['label']               for r in gt_rows]
    d['drift_mu_before'] = [float(r['mean_before'])  for r in gt_rows]
    d['drift_mu_after']  = [float(r['mean_after'])   for r in gt_rows]
    d['drift_sd_before'] = [float(r['std_before'])   for r in gt_rows]
    d['drift_sd_after']  = [float(r['std_after'])    for r in gt_rows]
    d['n_drifts']        = len(gt_rows)

    # ── Detection results ─────────────────────────────────────────────────────
    res_rows = load_csv(os.path.join(exp_dir, 'csv_05_detection_results.csv'))
    d['detected']         = [r['detected'] == 'True'    for r in res_rows]
    d['alarm_positions']  = [int(float(r['alarm_position']))
                              if _safe_float(r['alarm_position']) is not None else None
                              for r in res_rows]
    d['delays']           = [int(float(r['delay_samples']))
                              if _safe_float(r['delay_samples']) is not None else None
                              for r in res_rows]
    d['n_detected']       = sum(d['detected'])
    d['matched_alarms']   = [ap for ap in d['alarm_positions'] if ap is not None]

    # ── Raw stream ────────────────────────────────────────────────────────────
    stream_rows     = load_csv(os.path.join(exp_dir, 'csv_01_stream_raw.csv'))
    d['stream_idx'] = np.array([int(r['index'])   for r in stream_rows])
    d['stream_vals']= np.array([float(r['value']) for r in stream_rows])
    d['N']          = len(stream_rows)

    # Nominal σ per segment (initial + one per drift)
    d['seg_stds'] = [d['drift_sd_before'][0]] + d['drift_sd_after']

    # ── PDF snapshots ─────────────────────────────────────────────────────────
    snap_rows      = load_csv(os.path.join(exp_dir, 'csv_06_pdf_snapshots.csv'))
    snap_cols      = [c for c in snap_rows[0].keys() if c != 'x']
    d['snap_x']    = np.array([float(r['x']) for r in snap_rows])
    d['snap_cols'] = snap_cols
    d['snap_pdfs'] = {col: np.array([float(r[col]) for r in snap_rows])
                      for col in snap_cols}

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Framework Architecture
# ─────────────────────────────────────────────────────────────────────────────
def figure1_architecture(out_dir):
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    fig.patch.set_facecolor('white')

    C_STREAM='#1a5276'; C_IPNN='#1e8449'; C_DETECT='#922b21'
    C_OUT='#784212'; C_REF='#5b2c6f'; C_GREY='#566573'; C_ARROW='#2c3e50'

    def box(x, y, w, h, label, sublabel, color, fs=10):
        ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.08',
                     linewidth=1.6, edgecolor=color, facecolor=color+'22'))
        ax.text(x+w/2, y+h*0.62, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=color)
        ax.text(x+w/2, y+h*0.28, sublabel, ha='center', va='center',
                fontsize=7.5, color=C_GREY, style='italic')

    def arrow(x1,y1,x2,y2,label='',color=C_ARROW):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.6))
        if label:
            ax.text((x1+x2)/2,(y1+y2)/2+0.18,label,ha='center',
                    va='bottom',fontsize=7.5,color=color,style='italic')

    BH,BY,BW,G = 1.4,3.5,1.9,0.1
    box(0.3, BY,BW,BH,'Data Stream',       'xₙ, n=0,1,2,…',                      C_STREAM)
    box(2.6, BY,BW,BH,'IPNN Update',        'âⱼ(n)=(1−γₙ)âⱼ(n−1)\n+γₙ·φⱼ(xₙ)',C_IPNN)
    box(4.9, BY,BW,BH,'PDF Evaluator',      'f̂(xᵢ)=Σⱼ âⱼ·φⱼ(xᵢ)\n400 pts [−4,10]',C_IPNN)
    box(7.2, BY,BW,BH,'ISE Monitor',        'ISEₙ=∫(f̂cur−f̂ref)²dx\n(trapezoidal)',C_DETECT)
    box(9.5, BY,BW,BH,'Confirmation\nFilter','Counter ≥ c=7\nconsecutive steps',  C_DETECT)
    box(11.8,BY,BW,BH,'Drift Alarm',        'Record n*\nUpdate reference',         C_OUT)

    arrow(0.3+BW+G, BY+BH/2, 2.6-G,  BY+BH/2, 'xₙ')
    arrow(2.6+BW+G, BY+BH/2, 4.9-G,  BY+BH/2, 'â_j(n)')
    arrow(4.9+BW+G, BY+BH/2, 7.2-G,  BY+BH/2, 'f̂cur')
    arrow(7.2+BW+G, BY+BH/2, 9.5-G,  BY+BH/2, 'ISEₙ')
    arrow(9.5+BW+G, BY+BH/2,11.8-G,  BY+BH/2, 'alarm')

    RBY=1.4
    box(7.2,RBY,BW,BH,'Reference PDF','f̂ref captured\nat n=w=300',C_REF)
    arrow(7.2+BW/2,RBY+BH,7.2+BW/2,BY,'f̂ref',color=C_REF)
    ax.annotate('',xy=(7.2+BW/2,RBY+BH+0.05),xytext=(11.8+BW/2,RBY+BH+0.05),
                arrowprops=dict(arrowstyle='->',color=C_OUT,lw=1.4,
                                linestyle='dashed',connectionstyle='arc3,rad=-0.35'))
    ax.text(9.7,2.85,'Adaptive reference reset after alarm',
            ha='center',fontsize=8,color=C_OUT,style='italic')
    ax.annotate('',xy=(4.9+BW/2,RBY+BH),xytext=(4.9+BW/2,BY),
                arrowprops=dict(arrowstyle='->',color=C_REF,lw=1.2,linestyle='dotted'))
    ax.text(5.3,2.7,'Snapshot\nat n=300',ha='left',fontsize=7.5,color=C_REF,style='italic')

    ax.text(0.35,2.9,
            "Primary configuration\n"
            "─────────────────────\n"
            "  Basis : Hermite series\n"
            "  Q = 0.5,  k = 3.0,  γ = 1.0\n"
            "  q(n) = 3·(n+1)⁰·⁵\n"
            "  Warm-up  w = 300\n"
            "  Threshold τ = 0.08\n"
            "  Confirmation c = 7",
            ha='left',va='top',fontsize=8,family='monospace',color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4',facecolor='#eaf2ff',
                      edgecolor='#1a5276',linewidth=1.2))

    ax.set_title('Figure 2 – IPNN-Based Concept Drift Detection Framework',
                 fontsize=13,fontweight='bold',color='#1a252f',pad=12)
    save(fig,'Figure1_Architecture.png',out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Conceptual illustration
# ─────────────────────────────────────────────────────────────────────────────
def figure2_concept(out_dir):
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    x = np.linspace(-2.5,6.5,600)
    ax.plot(x,gauss(x,0.0,1.0), color='#1a5276',lw=2.2,label='Reference f̂_ref  (stable period)')
    ax.plot(x,gauss(x,1.25,1.3),color='#f39c12',lw=2.0,linestyle='--',label='Transitional f̂_cur  (drift onset)')
    ax.plot(x,gauss(x,2.5,1.0), color='#c0392b',lw=2.2,label='Post-drift f̂_cur  (new concept)')
    diff2=(gauss(x,2.5,1.0)-gauss(x,0.0,1.0))**2
    ax.fill_between(x,0,diff2*3.5,alpha=0.14,color='#c0392b',label='(f̂_cur − f̂_ref)²  [ISE integrand]')
    ax.axvline(0.0,color='#1a5276',lw=1.0,linestyle=':')
    ax.axvline(2.5,color='#c0392b',lw=1.0,linestyle=':')
    ax.annotate('',xy=(2.5,0.30),xytext=(0.0,0.30),
                arrowprops=dict(arrowstyle='<->',color='#7d3c98',lw=1.5))
    ax.text(1.25,0.33,'Δμ = 2.5',ha='center',fontsize=9,color='#7d3c98',fontweight='bold')
    ax.set_xlabel('x',fontsize=11); ax.set_ylabel('Density',fontsize=11)
    ax.set_title('(a)  PDF evolution around an abrupt mean drift',fontsize=11,fontweight='bold')
    ax.legend(fontsize=8.5,loc='upper right'); ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25); ax.spines[['top','right']].set_visible(False)

    ax2 = axes[1]
    np.random.seed(7)
    t=np.arange(400); ise_sim=np.random.exponential(0.007,400)
    drift_t,alarm_t=150,190
    for i in range(drift_t,400):
        ise_sim[i]+=0.11*min((i-drift_t)/55.0,1.0)*np.random.uniform(0.75,1.25)
    ise_sim=np.clip(ise_sim,0,None)
    ax2.plot(t,ise_sim,color='#1e8449',lw=1.5,zorder=2,label='ISE(n)')
    ax2.axhline(THRESHOLD,color='#c0392b',lw=1.7,linestyle='--',label=f'Threshold τ={THRESHOLD}')
    ax2.axvline(drift_t,color='#1a5276',lw=2.0,linestyle=':',zorder=3,label=f'True drift  (n={drift_t})')
    ax2.axvline(alarm_t,color='#e74c3c',lw=2.0,linestyle='-',zorder=3,label=f'Alarm fired (n={alarm_t})')
    ax2.axvspan(alarm_t-7,alarm_t,alpha=0.13,color='#e74c3c',zorder=1,label='Confirmation window (c=7)')
    ax2.annotate(f'Delay = {alarm_t-drift_t} samples',xy=(alarm_t,0.065),xytext=(alarm_t+25,0.075),
                 arrowprops=dict(arrowstyle='->',color='#7d3c98',lw=1.3),fontsize=9,color='#7d3c98',fontweight='bold')
    ax2.axvspan(0,50,alpha=0.08,color='#1a5276')
    ax2.text(25,0.13,'Warm-up\n(w=300*)',ha='center',fontsize=8,color='#1a5276',style='italic')
    ax2.text(300,0.01,'*axis compressed for illustration',fontsize=7,color='grey')
    ax2.set_xlabel('Stream position n  (relative)',fontsize=11)
    ax2.set_ylabel('ISE value',fontsize=11)
    ax2.set_title('(b)  ISE monitoring signal and drift alarm',fontsize=11,fontweight='bold')
    ax2.legend(fontsize=8.5,loc='upper left'); ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.25); ax2.spines[['top','right']].set_visible(False)

    fig.suptitle('Figure 1 – Conceptual Illustration of IPNN-Based Drift Detection',
                 fontsize=13,fontweight='bold',y=1.01)
    plt.tight_layout()
    save(fig,'Figure2_Concept.png',out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – ISE Monitoring Timeline
# ─────────────────────────────────────────────────────────────────────────────
def figure3_ise_timeline(d, out_dir):
    ise=d['ise']; N=d['N']
    matched=d['matched_alarms']
    unmatched=[a for a in d['all_alarms'] if a not in matched]

    fig,ax=plt.subplots(figsize=(16,5.5))
    fig.patch.set_facecolor('white')

    ax.plot(np.arange(N),ise,color='#2980b9',lw=0.8,alpha=0.85,zorder=2,label='ISE(n)')
    ax.axhline(THRESHOLD,color='#c0392b',lw=1.6,linestyle='--',zorder=3,
               label=f'Threshold τ = {THRESHOLD}')
    ax.axvspan(0,WARMUP,alpha=0.07,color='#1a5276',zorder=1)
    ax.text(WARMUP/2,ise.max()*0.95,f'Warm-up\n(n < {WARMUP})',
            ha='center',fontsize=8,color='#1a5276',style='italic',va='top')

    for pos,dtype in zip(d['drift_positions'],d['drift_types']):
        ax.axvline(pos,color=TYPE_COLORS.get(dtype,'#888'),lw=1.3,linestyle=':',alpha=0.8,zorder=4)
    for a in matched:
        ax.plot(a,ise[a]+ise.max()*0.03,marker='v',color='#27ae60',ms=7,zorder=6,clip_on=False)
    for a in unmatched:
        ax.plot(a,ise[a]+ise.max()*0.03,marker='x',color='#e67e22',ms=6,lw=1.5,zorder=6,clip_on=False)
    for i,(pos,dtype) in enumerate(zip(d['drift_positions'],d['drift_types'])):
        ax.text(pos,-0.009,f'D{i+1}',ha='center',va='top',fontsize=7.5,
                color=TYPE_COLORS.get(dtype,'#888'),fontweight='bold',
                transform=ax.get_xaxis_transform(),clip_on=False)

    seen=list(dict.fromkeys(d['drift_types']))
    handles=[
        mpatches.Patch(color='#2980b9',label='ISE(n)'),
        mpatches.Patch(color='#c0392b',label=f'Threshold τ={THRESHOLD}'),
        *[plt.Line2D([0],[0],color=TYPE_COLORS.get(t,'#888'),lw=1.3,linestyle=':',
                     label=f'{t.capitalize()} drift (true)') for t in seen],
        plt.Line2D([0],[0],marker='v',color='#27ae60',ms=8,lw=0,
                   label=f'Matched alarm (TP) — {len(matched)}'),
        plt.Line2D([0],[0],marker='x',color='#e67e22',ms=8,lw=1.5,
                   label=f'Unmatched alarm (FP) — {len(unmatched)}'),
    ]
    ax.legend(handles=handles,fontsize=8,loc='upper right',ncol=2,framealpha=0.9)
    ax.set_xlim(0,N); ax.set_ylim(-0.005,ise.max()*1.15)
    ax.set_xlabel('Stream position n',fontsize=11)
    ax.set_ylabel('ISE value',fontsize=11)
    ax.set_title(f'Figure 4 – ISE Monitoring Timeline: {d["n_drifts"]} Drift Events, '
                 f'N={N:,}   (Q=0.5, k=3, τ={THRESHOLD}, w={WARMUP})',
                 fontsize=12,fontweight='bold')
    ax.grid(alpha=0.2); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    save(fig,'Figure3_ISE_Timeline.png',out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Detection Results + Q sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def figure4_detection_results(d, out_dir):
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    fig.patch.set_facecolor('white')

    # ── Left: per-drift delay bars (from CSV) ─────────────────────────────────
    ax=axes[0]
    det_idx   =[i for i,det in enumerate(d['detected']) if det]
    delays_det=[d['delays'][i]         for i in det_idx]
    types_det =[d['drift_types'][i]    for i in det_idx]
    mu_b      =[d['drift_mu_before'][i] for i in det_idx]
    mu_a      =[d['drift_mu_after'][i]  for i in det_idx]
    sd_b      =[d['drift_sd_before'][i] for i in det_idx]
    sd_a      =[d['drift_sd_after'][i]  for i in det_idx]
    ylabels   =[f"D{i+1}  {types_det[k].capitalize()}\n"
                f"(μ:{mu_b[k]}→{mu_a[k]}, σ:{sd_b[k]}→{sd_a[k]})"
                for k,i in enumerate(det_idx)]
    colors_d  =[TYPE_COLORS.get(t,'#888') for t in types_det]

    bars=ax.barh(range(len(delays_det)),delays_det,
                 color=colors_d,edgecolor='white',height=0.65,zorder=2)
    max_d=max(delays_det) if delays_det else 1
    for i,(bar,delay) in enumerate(zip(bars,delays_det)):
        ax.text(delay+max_d*0.01,i,str(delay),va='center',ha='left',
                fontsize=8.5,color='#2c3e50',fontweight='bold')

    if len(delays_det)>1:
        mean_d  =sum(delays_det)/len(delays_det)
        median_d=sorted(delays_det)[len(delays_det)//2]
        off=max_d*0.01
        ax.axvline(mean_d,  color='#2c3e50',lw=1.2,linestyle='--',alpha=0.7)
        ax.axvline(median_d,color='#555',   lw=1.0,linestyle=':',alpha=0.6)
        ax.text(mean_d+off,  len(delays_det)-0.6, f'Mean = {mean_d:.0f}',  fontsize=8,color='#2c3e50',style='italic')
        ax.text(median_d+off,len(delays_det)-1.4, f'Median = {median_d}',fontsize=8,color='#555',   style='italic')

    ax.set_yticks(range(len(delays_det))); ax.set_yticklabels(ylabels,fontsize=8)
    ax.set_xlabel('Detection delay (samples)',fontsize=11)
    tpr=100*d['n_detected']/d['n_drifts'] if d['n_drifts'] else 0
    ax.set_title(f'(a)  Per-event detection delay\n'
                 f'({d["n_detected"]}/{d["n_drifts"]} detected — TPR={tpr:.0f}%)',
                 fontsize=10.5,fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=v,label=k.capitalize())
                        for k,v in TYPE_COLORS.items()],
              fontsize=8,loc='lower right')
    ax.set_xlim(0,max_d*1.2); ax.grid(axis='x',alpha=0.25)
    ax.spines[['top','right']].set_visible(False)

    # ── Right: Q sensitivity (multi-run constants) ────────────────────────────
    ax2=axes[1]
    q_labels  =[q[3] for q in Q_SENSITIVITY]
    q_detected=[q[2] for q in Q_SENSITIVITY]
    best=max(q_detected)
    q_colors  =['#1e8449' if det==best else '#aec6cf' for det in q_detected]
    bars2=ax2.bar(range(len(Q_SENSITIVITY)),q_detected,color=q_colors,
                  edgecolor='white',width=0.6,zorder=2)
    for i,(bar,det) in enumerate(zip(bars2,q_detected)):
        ax2.text(i,det+0.15,str(det),ha='center',va='bottom',fontsize=9,
                 fontweight='bold',color='#1e8449' if det==best else '#2c3e50')
    ax2.axhline(best,color='#c0392b',lw=1.4,linestyle='--',
                label=f'Best = {best} drifts detected')
    ax2.set_xticks(range(len(Q_SENSITIVITY))); ax2.set_xticklabels(q_labels,fontsize=8)
    ax2.set_ylabel('Drifts detected',fontsize=11); ax2.set_ylim(0,best+2.5)
    ax2.set_title('(b)  Effect of series growth exponent Q\n(τ=0.08, w=300 unless noted)',
                  fontsize=10.5,fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(axis='y',alpha=0.25)
    ax2.spines[['top','right']].set_visible(False)

    fig.suptitle('Figure 6 – Detection Performance: Per-Drift Delays and Q-Parameter Sensitivity',
                 fontsize=12,fontweight='bold',y=1.01)
    plt.tight_layout()
    save(fig,'Figure4_Detection_Results.png',out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – PDF Evolution  (auto-discovers columns from CSV)
# ─────────────────────────────────────────────────────────────────────────────
def figure5_pdf_evolution(d, out_dir):
    snap_cols=d['snap_cols']
    snap_pdfs=d['snap_pdfs']
    x_vals   =d['snap_x']

    # Priority: reference first, then post-drift (not +200), then pre-drift, then final
    def priority(col):
        if 'reference' in col:                         return (0, col)
        if 'post_drift' in col and '_200' not in col:  return (1, col)
        if 'post_drift' in col:                        return (2, col)
        if 'pre_drift'  in col:                        return (3, col)
        if 'final'      in col:                        return (4, col)
        return (5, col)

    selected=sorted(snap_cols,key=priority)[:8]

    palette   =['#1a5276','#2980b9','#1e8449','#27ae60',
                '#e67e22','#922b21','#7d3c98','#2c3e50']
    linestyles=['-','--','-','--','-','--','-','--']

    ncols=4
    nrows=(len(selected)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(16,4*nrows))
    fig.patch.set_facecolor('white')
    axes_flat=np.array(axes).flatten()

    for i,col in enumerate(selected):
        ax=axes_flat[i]
        pdf=snap_pdfs[col]
        color=palette[i%len(palette)]
        ls   =linestyles[i%len(linestyles)]
        ax.plot(x_vals,pdf,color=color,lw=2.0,linestyle=ls)
        ax.fill_between(x_vals,0,pdf,alpha=0.12,color=color)
        ax.set_xlim(x_vals[0],x_vals[-1]); ax.set_ylim(bottom=0)

        title=(col.replace('_reference',' (Reference)')
                  .replace('_pre_drift',' pre-Drift')
                  .replace('_post_drift',' post-Drift')
                  .replace('_200','+200')
                  .replace('_final',' (Final)')
                  .replace('n','n=',1))
        ax.set_title(title,fontsize=9,fontweight='bold',color=color)
        ax.set_xlabel('x',fontsize=9); ax.set_ylabel('Density',fontsize=9)
        ax.grid(alpha=0.2); ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
        peak_idx=int(np.argmax(pdf))
        ax.axvline(x_vals[peak_idx],color=color,lw=0.8,linestyle=':',alpha=0.6)
        ax.text(x_vals[peak_idx],pdf[peak_idx]*0.92,
                f' x={x_vals[peak_idx]:.2f}',fontsize=7.5,color=color)

    for j in range(len(selected),len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'Figure 5 – IPNN PDF Evolution Across Key Stream Positions\n'
                 f'(Q=0.5, k=3, γ=1.0, τ={THRESHOLD}, N={d["N"]:,})',
                 fontsize=12,fontweight='bold',y=1.01)
    plt.tight_layout()
    save(fig,'Figure5_PDF_Evolution.png',out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Stream Overview
# ─────────────────────────────────────────────────────────────────────────────
def figure6_stream_overview(d, out_dir):
    idx = d['stream_idx']
    vals = d['stream_vals']
    N = d['N']

    W = 200
    roll_mean = np.convolve(vals, np.ones(W) / W, mode='same')
    roll_std = np.array([
        vals[max(0, i - W // 2):i + W // 2].std()
        for i in range(len(vals))
    ])

    seg_boundaries = [0] + d['drift_positions'] + [N]
    seg_stds = d['seg_stds']
    max_std = max(seg_stds) if seg_stds else 1.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    fig.patch.set_facecolor('white')

    # ============================================================
    # Figure Title (Above Entire Figure)
    # ============================================================
    fig.suptitle(
        'Figure 3 – Synthetic Stream Overview with Drift Markers and Detection Alarms',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    fig.subplots_adjust(
        top=0.92,
        hspace=0.08
    )

    # ============================================================
    # Main Stream Plot
    # ============================================================
    THIN = max(1, N // 4000)

    ax1.plot(
        idx[::THIN],
        vals[::THIN],
        color='#aec6cf',
        lw=0.4,
        alpha=0.55,
        zorder=1
    )

    ax1.plot(
        idx,
        roll_mean,
        color='#2c3e50',
        lw=1.4,
        zorder=2,
        label=f'Rolling mean (W={W})'
    )

    ax1.fill_between(
        idx,
        roll_mean - roll_std,
        roll_mean + roll_std,
        alpha=0.18,
        color='#2c3e50',
        zorder=1,
        label='±1 rolling std'
    )

    # Drift markers
    for pos, dtype in zip(d['drift_positions'], d['drift_types']):
        ax1.axvline(
            pos,
            color=TYPE_COLORS.get(dtype, '#888'),
            lw=1.3,
            linestyle=':',
            alpha=0.85,
            zorder=3
        )

    # Detection alarms
    for alarm in d['matched_alarms']:
        ax1.axvline(
            alarm,
            color='#27ae60',
            lw=0.9,
            linestyle='-',
            alpha=0.45,
            zorder=3
        )

    ax1.set_ylabel(
        'Observation value',
        fontsize=11
    )

    ax1.grid(alpha=0.18)
    ax1.spines[['top', 'right']].set_visible(False)

    # ============================================================
    # Legend
    # ============================================================
    seen = list(dict.fromkeys(d['drift_types']))

    handles = [
        *[
            mpatches.Patch(
                color=TYPE_COLORS.get(t, '#888'),
                label=f'{t.capitalize()} drift'
            )
            for t in seen
        ],
        plt.Line2D(
            [0], [0],
            color='#27ae60',
            lw=1.2,
            label='Detected alarm'
        ),
        plt.Line2D(
            [0], [0],
            color='#2c3e50',
            lw=1.4,
            label='Rolling mean'
        )
    ]

    ax1.legend(
        handles=handles,
        fontsize=8.5,
        loc='upper right',
        ncol=3,
        framealpha=0.9
    )

    # ============================================================
    # Drift Labels
    # ============================================================
    for i, (pos, dtype) in enumerate(
        zip(d['drift_positions'], d['drift_types'])
    ):
        ax1.text(
            pos,
            1.01,
            f'D{i+1}',
            ha='center',
            fontsize=7.5,
            color=TYPE_COLORS.get(dtype, '#888'),
            fontweight='bold',
            transform=ax1.get_xaxis_transform(),
            clip_on=False
        )

    # ============================================================
    # Segment Variance Bar
    # ============================================================
    cmap = plt.cm.RdYlGn

    for j in range(len(seg_boundaries) - 1):
        a, b = seg_boundaries[j], seg_boundaries[j + 1]

        std_j = seg_stds[j] if j < len(seg_stds) else 1.0

        ax2.barh(
            0,
            b - a,
            left=a,
            height=0.5,
            color=cmap(
                1 - min(std_j / (max_std * 1.05), 1.0)
            ),
            edgecolor='white'
        )

        ax2.text(
            (a + b) / 2,
            0,
            f'σ={std_j}',
            ha='center',
            va='center',
            fontsize=7.5,
            fontweight='bold',
            color='#2c3e50'
        )

    ax2.set_yticks([])
    ax2.set_xlabel(
        'Stream position n',
        fontsize=11
    )

    ax2.set_xlim(0, N)

    ax2.spines[['top', 'right', 'left']].set_visible(False)

    ax2.set_title(
        'Nominal standard deviation per segment',
        fontsize=9,
        loc='left',
        color='#555'
    )

    # ============================================================
    # Save Figure
    # ============================================================
    save(
        fig,
        'Figure6_Stream_Overview.png',
        out_dir
    )
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser=argparse.ArgumentParser(
        description='Generate all thesis figures for the IPNN drift-detection research.')
    parser.add_argument('--exp_dir',default=DEFAULT_EXP,
        help='Path to a single experiment subfolder containing csv_0*.csv files.\n'
             'Example: experiments/20260415_160256_N20000_...')
    parser.add_argument('--out_dir',default=DEFAULT_OUT,
        help='Directory where PNG figures are saved (created if it does not exist).')
    args=parser.parse_args()

    exp_dir=os.path.abspath(args.exp_dir)
    out_dir=os.path.abspath(args.out_dir)

    if not os.path.isdir(exp_dir):
        print(f'ERROR: experiment folder not found:\n  {exp_dir}')
        print('Pass --exp_dir pointing to a subfolder inside experiments/, e.g.:')
        print('  python generate_figures.py --exp_dir "experiments/20260415_160256_N20000_..."')
        sys.exit(1)

    os.makedirs(out_dir,exist_ok=True)
    print(f'\nExperiment folder : {exp_dir}')
    print(f'Output folder     : {out_dir}')
    print(f'\nLoading experiment data...')
    d=load_experiment(exp_dir)
    print(f'  N={d["N"]:,}  |  drifts={d["n_drifts"]}  |  '
          f'detected={d["n_detected"]}/{d["n_drifts"]}  |  '
          f'alarms={len(d["all_alarms"])}')

    print(f'\nGenerating figures...')
    figure1_architecture(out_dir)
    figure2_concept(out_dir)
    figure3_ise_timeline(d,out_dir)
    figure4_detection_results(d,out_dir)
    figure5_pdf_evolution(d,out_dir)
    figure6_stream_overview(d,out_dir)
    print('\nAll 6 figures generated successfully.')


if __name__=='__main__':
    main()
