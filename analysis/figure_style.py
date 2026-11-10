"""Shared v10.5 manuscript style; visual encoding only, no statistical changes."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.text import Text

WIDTH = 160 / 25.4  # exact manuscript text width, in inches
COLORS = {
    'correction': '#009E73', 'harm': '#D55E00', 'net': '#263238',
    'accent': '#0072B2', 'absent': '#B8BEC4', 'grid': '#D9DEE2',
    'pale': '#EFF5F8', 'neutral': '#F2F3F4', 'line': '#54626C',
}
MODEL_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
LEGEND = dict(frameon=False, fontsize=8, handlelength=1.6,
              columnspacing=1.5, handletextpad=.5, borderaxespad=0)

def apply_style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':8.5,
        'axes.labelsize':8.5, 'axes.titlesize':9, 'axes.titleweight':'bold',
        'xtick.labelsize':7.5, 'ytick.labelsize':7.5, 'legend.fontsize':8,
        'text.color':COLORS['net'], 'axes.labelcolor':COLORS['net'],
        'axes.edgecolor':COLORS['line'], 'axes.linewidth':.7,
        'xtick.color':COLORS['net'], 'ytick.color':COLORS['net'],
        'axes.spines.top':False, 'axes.spines.right':False,
        'lines.linewidth':1.2, 'lines.markersize':4,
        'grid.color':COLORS['grid'], 'grid.linewidth':.55,
        'hatch.linewidth':.45, 'pdf.fonttype':42, 'ps.fonttype':42,
        'figure.facecolor':'white', 'savefig.facecolor':'white',
        'mathtext.fontset':'dejavusans', 'savefig.bbox':None,
    })

def save_figure(fig, root, name):
    """Keep all figures at 160 mm so the same font sizes survive insertion."""
    root=Path(root);fig.set_size_inches(WIDTH,fig.get_figheight(),forward=True)
    (root/'figures').mkdir(parents=True, exist_ok=True)
    (root/'documentation').mkdir(parents=True, exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.bbox.width,fig.bbox.height
    sizes=[];outside=[]
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text().strip():
            box=artist.get_window_extent(renderer)
            # Skip hidden tick labels returned by findobj on inactive axes.
            if box.width==0 or box.height==0:continue
            sizes.append(artist.get_fontsize())
            if box.x0 < -2 or box.y0 < -2 or box.x1 > w+2 or box.y1 > h+2:
                outside.append(artist.get_text())
    if outside:raise ValueError(f'{name}: text outside canvas: {outside}')
    for ext in ['pdf','png']:
        fig.savefig(root/'figures'/f'{name}.{ext}',dpi=300,
                    metadata={'Creator':'Matplotlib; shared IDA v10.5 figure style'} if ext=='pdf' else None)
    style=root/'documentation/figure_style_manifest.json'
    values=json.loads(style.read_text()) if style.exists() else {}
    values[name]={'width_mm':160,'height_mm':round(fig.get_figheight()*25.4,2),
        'font_family':'DejaVu Sans','font_min_pt':min(sizes),'font_max_pt':max(sizes),
        'legend_font_pt':8,'pdf':'vector','png_dpi':300,'clipped_text':outside}
    style.write_text(json.dumps(values,indent=2)+'\n')
    plt.close(fig)
