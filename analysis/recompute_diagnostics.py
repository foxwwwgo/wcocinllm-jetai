from pathlib import Path
import json, math, itertools, hashlib, csv
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import norm, beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]
rows=json.loads((R/'data/evaluation_rows.json').read_text())
a=json.loads((R/'data/recomputed_statistics.json').read_text())['initial']
names={'deepseek':'DeepSeek-V4.1-Flash','qwen':'qwen3.8-max-0902','glm':'glm-5.2','open_qwen14b':'Qwen2.5-14B-Instruct'}
models=list(names); tasks=['gsm8k','commonsenseqa']; labels=['GSM8K','CommonsenseQA']
betas=[0,1,2,4,8,16,32,64]
from figure_style import WIDTH, COLORS, MODEL_COLORS, LEGEND, apply_style, save_figure
apply_style()

# Additional, explicitly retrospective analyses. No new policy is selected.
group=defaultdict(list)
for r in rows:
 if r['model']!='open_qwen14b':group[r['id']].append(r)
out={'status':'retrospective sensitivity, not a replacement primary test','source_sha256':hashlib.sha256((R/'data/evaluation_rows.json').read_bytes()).hexdigest()}
out['sign_symmetry_sensitivity']={}
for key,fun in [('H1',lambda r:int(r['correct']['M']==1 and r['correct']['W64']==0)-int(r['correct']['M']==1 and r['correct']['W4']==0)),('H2',lambda r:r['correct']['G64']-r['correct']['W64'])]:
 # Equal task sizes: multiply all question means by 3 to work in integer units.
 vals=np.array([round(sum(fun(r) for r in rr)) for rr in group.values()],int)
 nz=vals[vals!=0];observed=abs(nz.sum())
 sums=np.array([sum(s*v for s,v in zip(signs,np.abs(nz))) for signs in itertools.product([-1,1],repeat=len(nz))])
 out['sign_symmetry_sensitivity'][key]={'nonzero_questions':len(nz),'integer_contrasts':nz.tolist(),'assignments':len(sums),'two_sided_p':float(np.mean(abs(sums)>=observed)),'assumption':'independent question signs jointly symmetric under the null; stronger than a mean-zero null; not a randomized-treatment test'}
ps=out['sign_symmetry_sensitivity'];order=sorted(ps,key=lambda k:ps[k]['two_sided_p']);last=0
for j,k in enumerate(order):last=max(last,min(1,(2-j)*ps[k]['two_sided_p']));ps[k]['holm_p']=last

out['gate_accounting']={}
for m in models:
 rr=[r for r in rows if r['model']==m]
 out['gate_accounting'][m]={'prevented_harms':sum(r['correct']['M']==1 and r['correct']['W64']==0 and r['correct']['G64']==1 for r in rr),'lost_corrections':sum(r['correct']['M']==0 and r['correct']['W64']==1 and r['correct']['G64']==0 for r in rr),'net_vs_majority':sum(r['correct']['G64']-r['correct']['M'] for r in rr)}

# Sensitivity calculations, not estimates of the power of an unobserved new rule.
zs=norm.ppf(1-.025/2)+norm.ppf(.80)
out['planning_grid']=[{'discordance':q,'target_gain':delta,'approx_n':math.ceil(zs**2*(q-delta**2)/delta**2)} for q in [.01,.02,.05,.10] for delta in [.0025,.005,.01] if delta<=q]
out['planning_note']='Normal paired-binary approximation, two-sided alpha .025, 80% power, q=P(D!=0). Not a cluster-adjusted sample-size guarantee; future model averages require pilot covariance and simulation.'
out['opportunity']={}
out['certificate']={}
for t in tasks:
 for m in models:
  rr=[r for r in rows if r['task']==t and r['model']==m];k=f'{t}:{m}'
  out['opportunity'][k]={'majority_wrong_gold_present':sum(r['correct']['M']==0 and r['gold'] in r['answers'] for r in rr),'majority_wrong_gold_absent':sum(r['correct']['M']==0 and r['gold'] not in r['answers'] for r in rr),'majority_correct':sum(r['correct']['M']==1 for r in rr),'constant_scores':sum(r['score_constant'] for r in rr)}
  cert={}
  for b in betas:
   safe=[]
   for r in rr:
    if not r['complete']:continue
    c=Counter(v for v in r['answers'] if v is not None); counts=sorted(c.values(),reverse=True)
    s=[s/100 for v,s in zip(r['answers'],r['scores']) if v is not None]
    ok=len(counts)==1 or (counts[0]>counts[1] and b*(max(s)-min(s))<math.log(counts[0]/counts[1]))
    if ok:
     assert r['predictions']['W'+str(b)]==r['majority'];safe.append(r)
   cert[str(b)]={'certified':len(safe),'denominator':sum(r['complete'] for r in rr)}
  out['certificate'][k]=cert
(R/'data/additional_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')

# Figure 1: policy paths and the common recovery-harm vocabulary.
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(WIDTH,1.85))
fig.subplots_adjust(left=.015,right=.985,bottom=.04,top=.98)
ax.set(xlim=(0,10),ylim=(0,3.2));ax.axis('off')
boxes=[(.05,1.5,2.05,1.18,'Same candidates\nand answer order\nfor all readouts','accent','pale'),
       (2.85,2.35,2.12,.58,'Majority M','line','neutral'),
       (2.85,1.18,2.12,.90,'Self-confidence\nFixed ranks; vary beta','accent','pale'),
       (5.72,1.48,4.15,1.45,'','accent','pale')]
for x,y,w,h,txt,edge,face in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',linewidth=.8,edgecolor=COLORS[edge],facecolor=COLORS[face]))
 if txt:ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8.5)
ax.text(7.80,2.65,'Same-question comparison',ha='center',va='center',fontsize=9,weight='bold')
ax.text(7.80,2.23,'Correction C: wrong → right',ha='center',va='center',fontsize=8.5,color=COLORS['correction'])
ax.text(7.80,1.79,'Harm H: right → wrong / abstain',ha='center',va='center',fontsize=8.5,color=COLORS['harm'])
for p,q,role in [((2.15,2.35),(2.80,2.64),'line'),((2.15,1.8),(2.80,1.63),'accent'),((5.02,2.64),(5.67,2.58),'line'),((5.02,1.63),(5.67,1.84),'accent')]:
 ax.annotate('',xy=q,xytext=p,arrowprops={'arrowstyle':'->','lw':.8,'color':COLORS[role]})
ax.text(5,.77,'Paired diagnosis: net gain = C − H',ha='center',fontsize=9,weight='bold')
ax.text(5,.20,'All answer changes = C + H + J; J: wrong → other wrong',ha='center',fontsize=8.5)
save_figure(fig,R,'scope')

# Figure 3: fixed concentration levels; colors identify models in this figure.
fig,axs=plt.subplots(2,2,figsize=(WIDTH,4.35),sharex=True)
for ti,t in enumerate(tasks):
 for mi,m in enumerate(models):
  pp=a['cells'][f'{t}:{m}']['policies']
  style=dict(marker=['o','s','^','D'][mi],linestyle=['-','--','-.',':'][mi],ms=3.5,color=MODEL_COLORS[mi])
  axs[ti,0].plot(range(1,8),[pp['W'+str(b)]['difference_from_M_pp'] for b in betas[1:]],label=names[m],**style)
  axs[ti,1].plot(range(1,8),[100*pp['W'+str(b)]['harms']/300 for b in betas[1:]],**style)
  for j in range(2):axs[ti,j].plot([0],[0],**style)
 for j in range(2):
  ax=axs[ti,j];ax.axhline(0,color=COLORS['line'],lw=.55);ax.grid(axis='y');ax.set_axisbelow(True)
  ax.set_title(f'({chr(97+ti*2+j)}) {labels[ti]}',loc='left');ax.set_xticks(range(8),betas);ax.axvline(.5,color=COLORS['grid'],ls=':',lw=.8)
 axs[ti,0].set_ylabel('Accuracy change vs M (pp)');axs[ti,1].set_ylabel('Harm / all questions (%)')
axs[1,0].set_xlabel('Concentration beta (fixed grid)');axs[1,1].set_xlabel('Concentration beta (fixed grid)')
handles,labs=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labs,loc='upper center',ncol=2,bbox_to_anchor=(.5,.995),**LEGEND)
fig.tight_layout(pad=.65,rect=[0,0,1,.89],h_pad=1.35,w_pad=1.0)
save_figure(fig,R,'concentration')

# Figure 2: primary corrections/harms and archived confidence intervals.
fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.45),gridspec_kw={'width_ratios':[1,1.1]})
pol=['W4','W64','G64'];c=[sum(a['cells'][f'{t}:{m}']['policies'][p]['corrections'] for t in tasks for m in models[:3]) for p in pol];h=[sum(a['cells'][f'{t}:{m}']['policies'][p]['harms'] for t in tasks for m in models[:3]) for p in pol]
x=np.arange(3)
axs[0].bar(x-.17,c,.33,color=COLORS['correction'],edgecolor=COLORS['net'],linewidth=.4,label='Corrections (C)')
axs[0].bar(x+.17,h,.33,color=COLORS['harm'],edgecolor=COLORS['net'],linewidth=.4,hatch='///',label='Harms (H)')
for xx,cc,hh in zip(x,c,h):
 axs[0].text(xx-.17,cc+.25,str(cc),ha='center',fontsize=8)
 axs[0].text(xx+.17,hh+.25,str(hh),ha='center',fontsize=8)
axs[0].set_xticks(x,pol);axs[0].set_ylabel('Model–question records / 1,800');axs[0].set_ylim(0,16)
axs[0].set_title('(a) Changes from majority',loc='left')
metrics=[a['commercial_pooled']['primary']['H1_harm64_minus_harm4'],a['commercial_pooled']['primary']['H2_accuracy_guard64_minus_64'],a['commercial_pooled']['policies_vs_M']['G64']]
for i,v in enumerate(metrics):
 mu=v['effect_pp'];lo,hi=v['ci95_pp']
 axs[1].errorbar(mu,2-i,xerr=[[mu-lo],[hi-mu]],fmt=['o','s','D'][i],capsize=3,color=[COLORS['harm'],COLORS['accent'],COLORS['net']][i])
axs[1].set_yticks([2,1,0],['H1 (primary)','H2 (primary)','G64 − M (desc.)'])
axs[1].axvline(0,color=COLORS['line'],ls='--',lw=.7);axs[1].set_ylim(-.6,2.6)
axs[1].set_xlim(-.75,1.25);axs[1].set_xticks([-.5,0,.5,1])
axs[1].set_xlabel('Effect (pp); 95% cluster CI');axs[1].set_title('(b) Paired effects',loc='left')
handles,labs=axs[0].get_legend_handles_labels()
fig.legend(handles,labs,loc='upper center',ncol=2,bbox_to_anchor=(.5,.99),**LEGEND)
fig.tight_layout(pad=.65,rect=[0,0,1,.88],w_pad=1.5)
save_figure(fig,R,'tradeoff')

# Figure 4: recoverable opportunities use the same green as corrections.
fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.9))
for j,t in enumerate(tasks):
 vals=[out['opportunity'][f'{t}:{m}'] for m in models];y=np.arange(4)
 axs[j].barh(y,[v['majority_wrong_gold_absent'] for v in vals],color=COLORS['absent'],edgecolor=COLORS['net'],linewidth=.4,label='Correct class absent')
 axs[j].barh(y,[v['majority_wrong_gold_present'] for v in vals],left=[v['majority_wrong_gold_absent'] for v in vals],color=COLORS['correction'],edgecolor=COLORS['net'],linewidth=.4,hatch='///',label='Correct class present')
 axs[j].set_yticks(y,[names[m] for m in models] if j==0 else ['']*4);axs[j].invert_yaxis()
 axs[j].set_title(f'({chr(97+j)}) {labels[j]}',loc='left');axs[j].set_xlabel('Majority errors / 300 questions');axs[j].set_xlim(0,50)
 for yy,v in zip(y,vals):axs[j].text(v['majority_wrong_gold_absent']+v['majority_wrong_gold_present']+.7,yy,'o='+str(v['majority_wrong_gold_present']),va='center',fontsize=7.5)
handles,labs=axs[1].get_legend_handles_labels()
fig.legend(handles,labs,loc='upper center',ncol=2,bbox_to_anchor=(.5,.99),**LEGEND)
fig.tight_layout(pad=.65,rect=[0,0,1,.88],w_pad=1.2)
save_figure(fig,R,'opportunity')

print('Recreated four figures with the common 160-mm style; data unchanged.')
