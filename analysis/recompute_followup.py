# Offline publication reproduction; no API calls or collection.
"""Generate publication tables/figure from completed, independently checked data."""
from pathlib import Path
import json, shutil, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER=Path(__file__).resolve().parents[1]
(PAPER/'sections').mkdir(parents=True, exist_ok=True)
(PAPER/'figures').mkdir(parents=True, exist_ok=True)
SOURCE=PAPER
def read(p): return json.loads(p.read_text())
lock=read(SOURCE/'data/successor_development_lock.json')
audit=read(SOURCE/'data/independent_decision_replay.json')
assert audit['decision']==lock['status']
rep=read(SOURCE/'data/replication_analysis.json')
floats=read(SOURCE/'data/original_float_sensitivity.json')
names={'deepseek':'DeepSeek-V4.1-Flash','qwen':'qwen3.8-max-0902','glm':'glm-5.2',
       'open_qwen14b':'Qwen2.5-14B-Instruct','granite':'Granite-3.3-2B-Instruct','phi':'Phi-3-mini-4k-instruct'}
tasks={'gsm8k':'GSM8K','commonsenseqa':'CSQA'}
selected=lock['selected'];chosen=selected['key']
rows=read(SOURCE/'data/development_rows.json')

# Preserve compact underlying data and code-free numeric summaries in the source package.
# All checked input records are supplied in data/.

metrics={}
for model in names:
    for task in tasks:
        rr=[r for r in rows if r['model']==model and r['task']==task]
        metrics[task+':'+model]={'n':len(rr),'M':sum(int(r['correct']['M']) for r in rr),
            'F':sum(r['correct'][chosen] for r in rr),'DEV':sum(int(r['correct']['DEV']) for r in rr),
            'F_corrects':sum(not r['correct']['M'] and r['correct'][chosen] for r in rr),
            'F_harms':sum(bool(r['correct']['M'] and not r['correct'][chosen]) for r in rr),
            'DEV_beta':lock['DEV'][task+':'+model]['beta'],
            'valid_seats':sum(r['candidate_n'] for r in rr),
            'empty_pools':sum(r['candidate_n']==0 for r in rr),
            'score_failure_pools':sum(r['candidate_n']>0 and not r['complete'] for r in rr),
            'constant_score_pools':sum(r['score_constant'] for r in rr),
            'recoverable_majority_errors':sum(not r['correct']['M'] and r['gold'] in r['answers'] for r in rr),
            'F_overrides':sum(r['predictions'][chosen]!=r['majority'] for r in rr)}
summary={'status':lock['status'],'selected':selected,'development_cells':metrics,
         'M_correct':sum(int(r['correct']['M']) for r in rows),
         'DEV_correct':sum(int(r['correct']['DEV']) for r in rows),
         'F_corrects':sum(not r['correct']['M'] and r['correct'][chosen] for r in rows),
         'F_harms':sum(bool(r['correct']['M'] and not r['correct'][chosen]) for r in rows),
         'development_format_counts':{
             'scheduled_seats':7*len(rows),'valid_seats':sum(r['candidate_n'] for r in rows),
             'empty_pools':sum(r['candidate_n']==0 for r in rows),
             'score_failure_pools':sum(r['candidate_n']>0 and not r['complete'] for r in rows),
             'constant_score_pools':sum(r['score_constant'] for r in rows)},
         'no_development_inference':'All development comparisons are selection/screening descriptions, not confirmatory tests.',
         'source_hashes':{relative:hashlib.sha256((SOURCE/'data'/Path(relative).name).read_bytes()).hexdigest() for relative in
             ['results/development_rows.json','protocols/successor_development_lock.json','results/replication_analysis.json']}}
(PAPER/'data/followup_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

# Display all twelve rules in the locked rank order, including unfavorable outcomes.
cs=lock['all_candidates_ranked'];y=np.arange(len(cs))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.1,4.25),gridspec_kw={'width_ratios':[1.15,1]},sharey=True)
labels=[f"({c['beta']}, {c['maximum_gap']}, {c['minimum_confidence_gap']:.2f})" for c in cs]
net=[c['gain_vs_M_pp'] for c in cs]
axs[0].barh(y,net,color=['#0072B2' if v>=0 else '#D55E00' for v in net],height=.62)
axs[0].scatter(net,y,s=13,color='#263238',zorder=3)
axs[0].axvline(0,color='.4',lw=.7)
axs[0].axvline(.5,color='#555555',ls='--',lw=.8,label='Screen target: 0.5 pp')
axs[0].set_yticks(y,labels);axs[0].invert_yaxis();axs[0].set_ylabel(r'Rule $(\beta,h,\eta)$ in locked rank order')
axs[0].set_xlabel('Development gain over M (pp)');axs[0].set_title('(a) Net accuracy change')
left=min(-.10,min(net)-.18);right=max(.75,max(net)+.23);axs[0].set_xlim(left,right)
for yy,c in enumerate(cs):
    xx=c['gain_vs_M_pp'];axs[0].annotate(f"{c['net_count_vs_M']:+d}",(xx,yy),xytext=(4 if xx>=0 else -4,0),textcoords='offset points',va='center',ha='left' if xx>=0 else 'right',fontsize=7.5)
axs[0].legend(loc='lower left',bbox_to_anchor=(0,-.25),frameon=False,fontsize=7.5)
for yy,c in enumerate(cs):
    rate=c['retention']
    if rate is None:
        axs[1].text(2,yy,'undefined (0 opportunities)',va='center',fontsize=7)
    else:
        axs[1].barh(yy,100*rate,height=.62,color='#009E73')
        axs[1].text(min(100*rate+2,103),yy,f"{c['retained_recoveries']}/{c['base_recoveries']}",va='center',fontsize=7.5)
axs[1].axvline(75,color='#555555',ls='--',lw=.8,label='Screen target: 75%')
axs[1].set_xlim(0,122);axs[1].set_xticks([0,25,50,75,100]);axs[1].set_xlabel('Retained W-beta corrections (%)')
axs[1].set_title('(b) Recovery retention');axs[1].legend(loc='lower left',bbox_to_anchor=(0,-.25),frameon=False,fontsize=7.5)
for ax in axs: ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(PAPER/'figures/followup_gate.pdf',bbox_inches='tight')
fig.savefig(PAPER/'figures/followup_gate.png',dpi=200,bbox_inches='tight');plt.close(fig)

# Cross-family table; two parsers change the candidate pools, so report both explicitly.
lines=[r'\begin{table}[!htbp]\centering\small',
 r'\caption{Effects of answer-format compatibility.}\label{tab:followup-replication}',
 r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}\toprule',
 r'Task & Model & Valid (\%) & $M$ (\%) & $W_{64}-M$ & $G_{64}-M$\\\midrule']
for task in tasks:
    for model in ['granite','phi']:
        k=task+':'+model;a=audit['replication_original_parser'][k];b=audit['replication_compatibility_parser'][k]
        pair=lambda x,z:f'{x:.1f} / {z:.1f}'
        def diff(c,p):return 100*(c['policies'][p]['correct']-c['policies']['M']['correct'])/c['n']
        lines.append(f"{tasks[task]} & {names[model]} & {pair(100*a['valid_seats']/a['seats'],100*b['valid_seats']/b['seats'])} & {pair(100*a['policies']['M']['correct']/a['n'],100*b['policies']['M']['correct']/b['n'])} & {diff(a,'W64'):+.2f} / {diff(b,'W64'):+.2f} & {diff(a,'G64'):+.2f} / {diff(b,'G64'):+.2f}"+r'\\')
lines.extend([r'\bottomrule\end{tabular*}', r'\par\smallskip\parbox{\textwidth}{\footnotesize \textit{Note:} Entries: evaluation / compatibility parser; 300 questions per cell. Valid: percentage of 2,100 scheduled seats. $M$: accuracy (\%); changes: percentage points. CSQA: CommonsenseQA. Parser variants use different usable pools.}', r'\end{table}'])
(PAPER/'sections/followup_replication_table.tex').write_text('\n'.join(lines)+'\n')

# Cell results are development descriptions, never individual confirmatory successes.
lines=[r'\begin{table*}[!t]\centering\small',
 r'\caption{New-question development screen: 100 questions per task and model. Valid is the percentage of 700 scheduled seats; empty counts abstaining pools. $F$ is the single globally selected successor. Accuracy is in percent; C/H are corrections/harms against $M$. DEV is selected within each development cell. These are in-sample screening results, not an independent efficacy test.}\label{tab:followup-development}',
 r'\begin{tabular}{@{}llrrrrrrr@{}}\toprule',r'Model & Task & Valid (\%) & Empty & $M$ & $F$ & DEV & DEV $\beta$ & $F$ C/H\\\midrule']
for model in names:
    for index,task in enumerate(tasks):
        c=metrics[task+':'+model]
        lines.append(f"{names[model] if index==0 else ''} & {tasks[task]} & {100*c['valid_seats']/700:.1f} & {c['empty_pools']} & {c['M']:.0f} & {c['F']:.0f} & {c['DEV']:.0f} & {c['DEV_beta']} & {c['F_corrects']}/{c['F_harms']}"+r'\\')
lines.extend([r'\bottomrule\end{tabular}\end{table*}'])
(PAPER/'sections/followup_development_table.tex').write_text('\n'.join(lines)+'\n')

lines=[r'\begin{table*}[!t]\centering\small',
 r'\caption{All twelve frozen development candidates, in the predefined ranking order. Net is the difference in correct model--question records from $M$ over 1,200 records. Task effects are percentage points. $Q$ counts distinct questions with at least one associated $W_\beta$ recovery; retention uses recovery records, not $Q$. The top row alone determines continuation under the precollection rule.}\label{tab:all-successor-candidates}',
 r'\begin{tabular}{@{}rrr rrr rrr@{}}\toprule',
 r'$\beta$ & $h$ & $\eta$ & Net & GSM8K & CSQA & Overrides & Retained/available & $Q$\\\midrule']
for c in cs:
    lines.append(f"{c['beta']} & {c['maximum_gap']} & {c['minimum_confidence_gap']:.2f} & {c['net_count_vs_M']:+d} & {c['task_gain_vs_M_pp']['gsm8k']:+.2f} & {c['task_gain_vs_M_pp']['commonsenseqa']:+.2f} & {c['overrides']} & {c['retained_recoveries']}/{c['base_recoveries']} & {c['distinct_recovery_opportunity_questions']}"+r'\\')
lines.extend([r'\bottomrule\end{tabular}\end{table*}'])
(PAPER/'sections/followup_candidate_table.tex').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
