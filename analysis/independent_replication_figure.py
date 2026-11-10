"""Replace Figure 2 with a same-scale original/independent-batch comparison."""
from pathlib import Path
import csv,json,math,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import WIDTH,COLORS,LEGEND,apply_style,save_figure
ROOT=Path(__file__).resolve().parents[1]
def main():
 assert (ROOT/'data/independent_replication/collection_complete.json').exists()
 original=json.loads((ROOT/'data/recomputed_statistics.json').read_text())['initial']['commercial_pooled']
 follow=json.loads((ROOT/'data/independent_replication/analysis.json').read_text())['primary_new_batch']
 oldrows=json.loads((ROOT/'data/evaluation_rows.json').read_text());oldrows=[r for r in oldrows if r['model'] in ['deepseek','qwen','glm']]
 newrows=json.loads((ROOT/'data/independent_replication/evaluation_rows.json').read_text())
 policies=['W4','W64','G64'];bar=[];points=[]
 for batch,rows,result in [('Original',oldrows,original),('Independent',newrows,follow)]:
  for p in policies:
   C=sum(r['correct']['M']==0 and r['correct'][p]==1 for r in rows);H=sum(r['correct']['M']==1 and r['correct'][p]==0 for r in rows)
   bar.append({'batch':batch,'policy':p,'C':C,'H':H,'net':C-H,'model_question_records':len(rows)})
  for endpoint,key in [('H1','H1_harm64_minus_harm4'),('H2','H2_accuracy_guard64_minus_64'),('G64-M','G64')]:
   v=result['policies_vs_M'][key] if endpoint=='G64-M' else result['primary'][key]
   points.append({'batch':batch,'endpoint':endpoint,'role':'descriptive' if endpoint=='G64-M' else 'primary','effect_pp':v['effect_pp'],'ci_low_pp':v['ci95_pp'][0],'ci_high_pp':v['ci95_pp'][1],'independent_questions':600})
 for name,rows in [('figure_2_recovery_harm_batches.csv',bar),('figure_2_effects_batches.csv',points)]:
  with (ROOT/'figure_data'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 apply_style();fig,axs=plt.subplots(2,2,figsize=(WIDTH,4.65),gridspec_kw={'width_ratios':[1,1.14]})
 top=max(max(v['C'],v['H']) for v in bar);ytop=max(5,math.ceil(top*1.24))
 lo=min(v['ci_low_pp'] for v in points);hi=max(v['ci_high_pp'] for v in points);span=max(.5,hi-lo);limits=(min(lo,0)-.15*span,max(hi,0)+.15*span)
 for bi,batch in enumerate(['Original','Independent']):
  ax=axs[bi,0];bb=[v for v in bar if v['batch']==batch];x=np.arange(3)
  ax.bar(x-.17,[v['C'] for v in bb],.33,color=COLORS['correction'],edgecolor=COLORS['net'],linewidth=.4,label='Corrections (C)')
  ax.bar(x+.17,[v['H'] for v in bb],.33,color=COLORS['harm'],edgecolor=COLORS['net'],linewidth=.4,hatch='///',label='Harms (H)')
  for j,v in enumerate(bb):
   ax.text(j-.17,v['C']+.018*ytop,str(v['C']),ha='center',fontsize=8)
   ax.text(j+.17,v['H']+.018*ytop,str(v['H']),ha='center',fontsize=8)
  ax.set(xticks=x,xticklabels=policies,ylim=(0,ytop),ylabel='Changes / 1,800 records')
  ax.set_yticks(np.arange(0,ytop+1,max(1,math.ceil(ytop/4))))
  ax.set_title(f'({chr(97+bi*2)}) {batch}: C/H',loc='left')
  ax=axs[bi,1]
  for i,v in enumerate(p for p in points if p['batch']==batch):
   mu=v['effect_pp'];a,b=v['ci_low_pp'],v['ci_high_pp']
   ax.errorbar(mu,2-i,xerr=[[mu-a],[b-mu]],fmt=['o','s','D'][i],capsize=3,color=[COLORS['harm'],COLORS['accent'],COLORS['net']][i])
  ax.axvline(0,color=COLORS['line'],ls='--',lw=.7)
  ax.set(yticks=[2,1,0],yticklabels=[r'$\Delta_H$ (harm)',r'$\Delta_{\rm gap}$ (accuracy)','G64 - M (desc.)'],ylim=(-.6,2.6),xlim=limits,xlabel='Effect (pp); 95% cluster CI')
  ax.set_xticks([v for v in ax.get_xticks() if limits[0]<=v<=limits[1]])
  ax.set_title(f'({chr(98+bi*2)}) {batch}: effects',loc='left')
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.995),**LEGEND)
 fig.tight_layout(pad=.65,rect=[0,0,1,.94],h_pad=1.45,w_pad=1.2);save_figure(fig,ROOT,'tradeoff')
 manifest=json.loads((ROOT/'figure_data/manifest.json').read_text())
 manifest['version']='10.15'
 for name in ['data/independent_replication/analysis.json','data/independent_replication/evaluation_rows.json']:
  manifest['source_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
 manifest['csv_rows']={p.name:sum(1 for _ in csv.DictReader(p.open())) for p in sorted((ROOT/'figure_data').glob('*.csv'))}
 manifest['verification']=manifest['verification'].split(' Current Figure 2 has')[0]+' Current Figure 2 has original and independent-generation rows, separately recomputed from their own 600 question clusters; the older Figure 2a/2b CSV files are retained as original-only provenance.'
 (ROOT/'figure_data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Figure 2: original and independent generation, common axes, four panels.')
if __name__=='__main__':main()
