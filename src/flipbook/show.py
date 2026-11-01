"""Notebook embed: render a PairReport as HTML inside IPython.
"""


import html

from flipbook.stats import compare
from flipbook.store import Store


def show(store: Store, run_a: str, run_b: str):
    pair = compare(store, run_a, run_b)
    rows = "".join(
        f"<tr><td>{html.escape(f['row_id'])}</td><td>{f['kind']}"
        f"{' (hard)' if f['hard'] else ''}</td>"
        f"<td>{f['p_a']:.2f}</td><td>{f['p_b']:.2f}</td></tr>"
        for f in pair.flips
    )
    a = pair.agreement
    body = f"""
    <b>{html.escape(run_a)}</b> acc={pair.acc_a:.3f} &nbsp;vs&nbsp;
    <b>{html.escape(run_b)}</b> acc={pair.acc_b:.3f}<br>
    &Delta; {pair.delta:+.3f}
    CI [{pair.delta_ci[0]:+.3f}, {pair.delta_ci[1]:+.3f}]
    on {pair.n_pairs} paired rows<br>
    both-right {a['both_right']} &middot; both-wrong {a['both_wrong']}
    &middot; a-only {a['a_only']} &middot; b-only {a['b_only']}<br>
    truncation {pair.truncation_rate_a:.2f} &rarr; {pair.truncation_rate_b:.2f}
    &middot; ${pair.cost_a:.3f} vs ${pair.cost_b:.3f}
    <table><tr><th>row</th><th>kind</th><th>p_a</th><th>p_b</th></tr>{rows}</table>
    """
    from IPython.display import HTML
    return HTML(body)
