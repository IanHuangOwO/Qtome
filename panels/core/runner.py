"""Runs the selected panels: compute() then draw(); one failing panel prints its traceback and the rest still run.
--redraw skips compute()."""
import traceback

from panels.core import registry


def run(stage, names, ctx):
    """Returns the panel names that failed."""
    failed = []
    for n in names:
        print(f'\n=== panel {n} ===', flush=True)
        try:
            m = registry.load(stage, n)
            if not ctx.redraw and hasattr(m, 'compute'):
                m.compute(ctx)
            if hasattr(m, 'draw'):
                m.draw(ctx)
            elif ctx.redraw:
                print('  (no figures: nothing to redraw)')
        except Exception:
            traceback.print_exc()
            failed.append(n)
    if failed:
        print(f'\nFAILED panels: {failed}')
    return failed
