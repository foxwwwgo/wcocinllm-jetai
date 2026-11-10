"""One-command offline replay, fresh statistics, parser diagnostics and figures."""
from pathlib import Path
import subprocess,sys,os,tempfile
ROOT=Path(__file__).resolve().parent
env=dict(os.environ)
env.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'ida-figure-cache'))
scripts=['verify_followup_decisions.py','verify_primary_and_cisc.py',
         'recompute_statistics.py','verify_parser_mapping.py','recompute_parser_audit.py',
         'recompute_mechanism_diagnostics.py','recompute_extended_diagnostics.py','recompute_diagnostics.py',
         'recompute_followup.py','create_six_model_overview.py',
         'confidence_diagnostics.py','score_quality_figure.py','export_figure_data.py',
         'recompute_independent_replication.py','independent_replication_figure.py',
         'verify_response_exports.py','replication_sensitivity.py']
for script in scripts:
    print('Running '+script,flush=True)
    subprocess.run([sys.executable,str(ROOT/script)],check=True,env=env)
print('Completed all offline analyses. No model or network requests were made.')
