"""Recreate the six-model recovery-harm figure from frozen evaluation rows.

No collection, model calls, re-fitting, or policy selection. All 48 displayed
model/parser/task/policy cells have 300 scheduled questions, including failures.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figure_style import WIDTH, COLORS, LEGEND, apply_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'deepseek': 'DeepSeek-V4.1-Flash',
    'qwen': 'qwen3.8-max-0902',
    'glm': 'glm-5.2',
    'open_qwen14b': 'Qwen2.5-14B-Instruct',
    'granite': 'Granite-3.3-2B-Instruct',
    'phi': 'Phi-3-mini-4k-instruct',
}
POLICIES = ['W4', 'W64', 'G64']
TASKS = ['gsm8k', 'commonsenseqa']
BLOCKS = [
    ('initial', 'evaluation_rows.json', list(MODELS)[:4],
     'Initial study: four configurations'),
    ('added_original', 'replication_rows.json', list(MODELS)[4:],
     'Added families: evaluation parser'),
    ('added_compatibility', 'replication_extended_parser_rows.json', list(MODELS)[4:],
     'Added families: compatibility parser'),
]

def collect():
    values = []
    question_ids = {}
    hashes = {}
    for block, file, models, _ in BLOCKS:
        path = ROOT/'data'/file
        hashes[file] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = json.loads(path.read_text())
        assert len(rows) == len(models)*600
        for model in models:
            for task in TASKS:
                rr = [r for r in rows if r['model'] == model and r['task'] == task]
                assert len(rr) == len({r['id'] for r in rr}) == 300
                ids = {r['id'] for r in rr}
                if task in question_ids:
                    assert ids == question_ids[task]
                question_ids[task] = ids
                for policy in POLICIES:
                    # Recompute correctness from decisions and gold, then compare
                    # with the archived indicators and the C-H identity.
                    outcomes = []
                    for r in rr:
                        m = r['predictions']['M'] == r['gold']
                        a = r['predictions'][policy] == r['gold']
                        assert m == bool(r['correct']['M'])
                        assert a == bool(r['correct'][policy])
                        outcomes.append((m, a))
                    c = sum(not m and a for m, a in outcomes)
                    h = sum(m and not a for m, a in outcomes)
                    net = sum(int(a)-int(m) for m, a in outcomes)
                    assert net == c-h
                    values.append(dict(block=block, model=model, model_name=MODELS[model],
                        task=task, policy=policy, scheduled_questions=len(rr),
                        corrections=c, harms=h, net=net, net_pp=100*net/len(rr)))
    assert len(values) == 48
    # Cross-check all plotted values against the full released readout grid.
    aliases = {'initial':'initial', 'added_original':'replication_original',
               'added_compatibility':'replication_compatibility'}
    with (ROOT/'tables/all_readouts.csv').open() as f:
        full = {(r['study'],r['model'],r['task'],r['policy']):r for r in csv.DictReader(f)}
    for v in values:
        expected = full[aliases[v['block']],v['model'],v['task'],v['policy']]
        for key in ['corrections','harms','net']:
            assert v[key] == float(expected[key])
    return values, hashes

def plot(values):
    apply_style()
    colors = {'C':COLORS['correction'],'H':COLORS['harm'],'net':COLORS['net']}
    fig, axes = plt.subplots(3,2,figsize=(WIDTH,6.5),
        gridspec_kw={'height_ratios':[15,7,7]})
    fig.subplots_adjust(left=.29,right=.99,top=.875,bottom=.065,hspace=.82,wspace=.18)
    lookup = {(v['block'],v['model'],v['task'],v['policy']):v for v in values}
    for row,(block,_,models,title) in enumerate(BLOCKS):
        for col,task in enumerate(TASKS):
            ax = axes[row,col]
            ys=[]; labels=[]
            for j,model in enumerate(models):
                offset=4*j
                if j%2 == 0: ax.axhspan(offset-.45,offset+2.45,color=COLORS['neutral'],zorder=0)
                for k,policy in enumerate(POLICIES):
                    y=offset+k; ys.append(y);labels.append(policy)
                    v=lookup[block,model,task,policy]
                    c,h,net=v['corrections'],v['harms'],v['net']
                    ax.barh(y,-h,height=.55,color=colors['H'],edgecolor=COLORS['net'],hatch='///',lw=.35,zorder=2)
                    ax.barh(y,c,height=.55,color=colors['C'],edgecolor=COLORS['net'],lw=.35,zorder=2)
                    ax.scatter(net,y,marker='D',s=12,color=colors['net'],edgecolor='white',linewidth=.35,zorder=4)
                    if h: ax.text(-h-.22,y,f'-{h}',ha='right',va='center',fontsize=7)
                    if c: ax.text(c+.20,y,f'+{c}',ha='left',va='center',fontsize=7)
                    ax.text(8.2,y,f'{net:+d}' if net else '0',ha='center',va='center',fontsize=7,color=colors['net'])
                if col == 0:
                    ax.text(-.185,offset+1,MODELS[model],transform=ax.get_yaxis_transform(),
                            ha='right',va='center',fontsize=7.5)
            ax.set_yticks(ys,labels if col==0 else ['']*len(ys)); ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y',length=0,pad=4)
            ax.set_xlim(-11.9,9.25)
            ax.set_ylim(4*(len(models)-1)+2.7,-.8)
            ax.set_xticks([-10,-5,0,5])
            ax.axvline(0,color='#69737B',lw=.65,zorder=1)
            ax.axvline(6.65,color='#D3D9DD',lw=.55,zorder=1)
            ax.grid(axis='x',alpha=.16,zorder=0)
            ax.text(8.2,-1.06,'Net',ha='center',va='bottom',fontsize=7,fontweight='bold')
            ax.set_title(f"({chr(97+row*2+col)}) {'GSM8K' if col==0 else 'CommonsenseQA'}",loc='left',pad=17)
            if row==2: ax.set_xlabel('Changes from majority (questions)')
        pos=axes[row,0].get_position()
        fig.text(.015,pos.y1+.065,title,fontsize=8.5,fontweight='bold',ha='left',va='bottom',color=COLORS['accent'])
    handles=[Patch(facecolor=colors['C'],edgecolor=COLORS['net'],linewidth=.4,label='Corrections (C)'),
             Patch(facecolor=colors['H'],edgecolor=COLORS['net'],linewidth=.4,hatch='///',label='Harms (H)'),
             Line2D([],[],marker='D',ls='',color=colors['net'],markersize=4,label='Net (C - H)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),
               ncol=3,**LEGEND)
    save_figure(fig,ROOT,'six_model_recovery_harm')

def main():
    for directory in ['figures','tables','data']:(ROOT/directory).mkdir(exist_ok=True)
    values,hashes=collect()
    with (ROOT/'tables/six_model_recovery_harm.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (ROOT/'data/six_model_recovery_harm_manifest.json').write_text(json.dumps({
        'purpose':'Descriptive cross-configuration recovery-harm overview',
        'source_sha256':hashes,'cells':48,'questions_per_cell':300,
        'policies':POLICIES,'net_definition':'corrections minus harms',
        'matching':'Identical question IDs within task across all configurations and both added-family parsers',
        'verified':'Recomputed from decisions/gold and matched every displayed count to all_readouts.csv',
        'sampling':'Shared questions; parser variants and model rows are not independent replications',
    },indent=2)+'\n')
    plot(values)
    print('Created six-model overview: 48 verified policy cells, 300 questions each.')

if __name__=='__main__':main()
