"""Figure 4: opportunity, within-question discrimination and calibration."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_style import WIDTH,COLORS,MODEL_COLORS,apply_style,save_figure
R=Path(__file__).resolve().parents[1];apply_style()
rows=json.loads((R/'data/evaluation_rows.json').read_text());quality=json.loads((R/'data/confidence_quality.json').read_text())['cells']
models=['deepseek','qwen','glm','open_qwen14b'];names=['DeepSeek-V4.1-Flash','qwen3.8-max-0902','glm-5.2','Qwen2.5-14B-Instruct'];letters=list('ABCD');tasks=['gsm8k','commonsenseqa']
fig,axes=plt.subplots(3,2,figsize=(WIDTH,6.0),gridspec_kw={'height_ratios':[1,1.15,1.2]})
fig.subplots_adjust(left=.09,right=.97,bottom=.08,top=.855,hspace=.57,wspace=.3)
markers=['o','s','^','D']
handles=[Line2D([0],[0],marker=mk,color=c,lw=0,label=f'{a}: {n}') for a,n,c,mk in zip(letters,names,MODEL_COLORS,markers)]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False,fontsize=8,handletextpad=.25,columnspacing=1)
for col,task in enumerate(tasks):
 title=['GSM8K','CommonsenseQA'][col];ax=axes[0,col]
 for i,m in enumerate(models):
  rr=[r for r in rows if r['model']==m and r['task']==task];o=sum(r['correct']['M']==0 and r['gold'] in r['answers'] for r in rr);missing=sum(r['correct']['M']==0 and r['gold'] not in r['answers'] for r in rr)
  ax.barh(i,o,color=COLORS['correction'],hatch='///',height=.57);ax.barh(i,missing,left=o,color=COLORS['absent'],height=.57)
  ax.text(o+missing+.25,i,f'{o} / {o+missing}',va='center',fontsize=7)
 ax.set_title(f'({"ab"[col]}) {title}: opportunities',fontsize=8.5);ax.set_yticks(range(4),letters);ax.invert_yaxis();ax.set_xlim(0,14 if col==0 else 55);ax.set_xticks([0,5,10] if col==0 else [0,20,40]);ax.set_xlabel('Recoverable / all majority errors',fontsize=8);ax.xaxis.grid(True);ax.set_axisbelow(True)
 ax=axes[1,col];ax.axvline(.5,color=COLORS['line'],ls='--',lw=.8)
 for i,m in enumerate(models):
  v=quality[f'initial_evaluation:{m}:{task}'];mean=v['wqd'];lo,hi=v['wqd_ci95'];ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt=markers[i],color=MODEL_COLORS[i],capsize=2)
  ax.text(.98,i,f'n={v["wqd_questions"]}',ha='right',va='center',fontsize=7)
 ax.set_title(f'({"cd"[col]}) Within-question discrimination',fontsize=8.5);ax.set_xlim(.2,1);ax.set_yticks(range(4),letters);ax.invert_yaxis();ax.set_xlabel('WQD with 95% question-bootstrap interval',fontsize=7.5);ax.set_xticks([.25,.5,.75,1]);ax.xaxis.grid(True);ax.set_axisbelow(True)
 ax=axes[2,col];ax.plot([0,1],[0,1],ls='--',color=COLORS['line'],lw=.8)
 for i,m in enumerate(models):
  bins=[b for b in quality[f'initial_evaluation:{m}:{task}']['reliability'] if b['candidates']]
  ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],color=MODEL_COLORS[i],lw=.8,alpha=.8,ls=['-','--','-.',':'][i])
  ax.scatter([b['confidence'] for b in bins],[b['accuracy'] for b in bins],s=[9+24*np.sqrt(b['question_mass']/300) for b in bins],color=MODEL_COLORS[i],marker=markers[i],zorder=3)
 ax.set_title(f'({"ef"[col]}) Confidence and correctness',fontsize=8.5);ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='Mean confidence',ylabel='Correct fraction');ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.grid(True);ax.set_axisbelow(True)
save_figure(fig,R,'score_quality_grid')
