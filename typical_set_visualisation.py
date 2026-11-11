"""Typical set vs most-likely set for a length-10 Bernoulli source.

Run: python typical_set_visualisation.py
Requires: numpy and matplotlib.
"""
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


N = 10
N_SAMPLES = 150
rng = np.random.default_rng(2026)
all_sequences = np.asarray(list(product((0, 1), repeat=N)), dtype=np.int8)
ones_counts = all_sequences.sum(axis=1)
binary_weights = 1 << np.arange(N - 1, -1, -1)
sequence_codes = all_sequences @ binary_weights

COLORS = {
    'both': '#7950a1',
    'typical': '#00a6a6',
    'most_likely': '#ed8b00',
    'neither_edge': '#aeb8c4',
    'ink': '#172b4d',
}

fig = plt.figure(figsize=(13.5, 8.5), constrained_layout=False)
fig.suptitle('Typical sequences and most-likely sequences', fontsize=17, y=.975)

# Left side: definitions and separate controls for the two epsilons.
fig.text(.035, .91, 'Bernoulli source; sequence length n = 10', fontsize=10.5, weight='bold')
ax_p = fig.add_axes([.115, .865, .22, .025])
p_slider = Slider(ax_p, r'$p=P(X=1)$', .05, .95, valinit=.90, valstep=.05)
fig.text(.035, .835,
         r'$H(p)=-p\log_2p-(1-p)\log_2(1-p)$ bits/symbol', fontsize=9.5)

fig.text(.035, .78, 'Typical set', fontsize=11, weight='bold', color=COLORS['typical'])
fig.text(.035, .745,
         r'$T_n^{\epsilon_T}=\{x^n:| -n^{-1}\log_2P(x^n)-H(p)|\leq\epsilon_T\}$',
         fontsize=10.5)
ax_eps_t = fig.add_axes([.14, .685, .20, .022])
eps_t_slider = Slider(ax_eps_t, r'Tolerance $\epsilon_T$', .02, .70,
                      valinit=.20, valstep=.01)
fig.text(.035, .635,
         r'For sufficiently large $n$, $P(T_n^{\epsilon_T})\geq1-\epsilon_T$:' '\n'
         r'$(1-\epsilon_T)2^{n(H-\epsilon_T)}\leq|T_n^{\epsilon_T}|\leq2^{n(H+\epsilon_T)}$',
         fontsize=9.5, va='top', linespacing=1.3)

fig.text(.035, .48, 'Most-likely set', fontsize=11, weight='bold', color=COLORS['most_likely'])
fig.text(.035, .445,
         r'$S_n^{\epsilon_S}$: smallest set with $P(S_n^{\epsilon_S})\geq1-\epsilon_S$' '\n'
         'Take the highest-probability sequences first.',
         fontsize=10.5, va='top', linespacing=1.3)
ax_eps_s = fig.add_axes([.14, .35, .20, .022])
eps_s_slider = Slider(ax_eps_s, r'Mass outside $\epsilon_S$', .05, .95,
                      valinit=.10, valstep=.01)
fig.text(.035, .30,
         r'For sufficiently large $n$:' '\n'
         r'$(1-\epsilon_T-\epsilon_S)2^{n(H-\epsilon_T)}\leq|S_n^{\epsilon_S}|\leq2^{n(H+\epsilon_T)}$' '\n'
         r'If $\epsilon_T=\epsilon_S=\epsilon$, the lower factor is $1-2\epsilon$.',
         fontsize=9.0, va='top', linespacing=1.3)

fig.text(.035, .145,
         'Bounds are asymptotic; n = 10 statistics are exact.\n'
         'The S lower bound is useful when $\\epsilon_T+\\epsilon_S<1$.\n'
         'Ties put the displayed mode first, then use sequence order.',
         fontsize=8.3, va='top', linespacing=1.3, color='#526174')
ax_resample = fig.add_axes([.055, .02, .24, .05])
resample_button = Button(ax_resample, 'Draw new sequences')

# Right side: representative sequences and draws grouped by their number of 1s.
ax_examples = fig.add_axes([.45, .63, .52, .30])
ax_examples.set_axis_off()
ax_samples = fig.add_axes([.45, .12, .52, .43])
sample_state = {'values': rng.binomial(1, .90, size=(N_SAMPLES, N))}


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def sequence_probability_by_count(p):
    return p ** ones_counts * (1 - p) ** (N - ones_counts)


def membership_label(in_typical, in_most_likely):
    if in_typical and in_most_likely:
        return 'in both sets'
    if in_typical:
        return 'typical only'
    if in_most_likely:
        return 'most-likely only'
    return 'in neither set'


def update(_=None):
    p = float(p_slider.val)
    eps_t = float(eps_t_slider.val)
    eps_s = float(eps_s_slider.val)
    h = entropy(p)
    # A deterministic representative of the most-probable sequence. For p=1/2
    # every sequence is tied, so put this representative first in the tie order.
    mode_bit = int(p >= .5)
    mode_sequence = np.full(N, mode_bit, dtype=np.int8)
    mode_code = int(mode_sequence @ binary_weights)

    # For a Bernoulli-i.i.d. source, a sequence's probability depends only on
    # its number of 1s. Typicality is defined using per-symbol self-information.
    info_by_count = -(np.arange(N + 1) * np.log2(p)
                      + (N - np.arange(N + 1)) * np.log2(1 - p)) / N
    typical_by_count = np.abs(info_by_count - h) <= eps_t + 1e-12
    typical_mask = typical_by_count[ones_counts]

    # The most-likely set is the minimum-cardinality set with mass >= 1-eps_s.
    # Equal-probability strings are ordered lexicographically for reproducibility.
    sequence_probabilities = sequence_probability_by_count(p)
    tie_order = sequence_codes.copy()
    tie_order[mode_code] = -1
    order = np.lexsort((tie_order, -sequence_probabilities))
    cumulative_mass = np.cumsum(sequence_probabilities[order])
    cutoff = int(np.searchsorted(cumulative_mass, 1 - eps_s, side='left'))
    most_likely_mask = np.zeros(len(all_sequences), dtype=bool)
    most_likely_mask[order[:cutoff + 1]] = True

    # The most probable string uses the individually more probable symbol.
    # The most typical string has an empirical frequency nearest to p.
    typical_count = int(np.argmin(np.abs(info_by_count - h)))
    typical_sequence = np.r_[np.ones(typical_count, dtype=np.int8),
                             np.zeros(N - typical_count, dtype=np.int8)]

    def info_of(seq):
        k = int(seq.sum())
        return float(info_by_count[k])

    def probability_of(seq):
        return float(p ** int(seq.sum()) * (1 - p) ** (N - int(seq.sum())))

    typical_code = int(typical_sequence @ binary_weights)
    mode_is_typical = bool(typical_mask[mode_code])
    mode_is_most_likely = bool(most_likely_mask[mode_code])
    typ_is_typical = bool(typical_mask[typical_code])
    typ_is_most_likely = bool(most_likely_mask[typical_code])

    ax_examples.clear()
    ax_examples.set_axis_off()
    ax_examples.text(.00, .98, 'Two useful examples', transform=ax_examples.transAxes,
                     fontsize=11, weight='bold', color=COLORS['ink'], va='top')
    ax_examples.text(.02, .78,
                     'Most likely (highest-probability) sequence',
                     transform=ax_examples.transAxes, fontsize=9.5, va='center')
    ax_examples.text(.58, .78, ''.join(map(str, mode_sequence)),
                     transform=ax_examples.transAxes, fontsize=13, family='monospace',
                     weight='bold', color=COLORS['most_likely'], va='center')
    ax_examples.text(.02, .58,
                     'Most typical (self-information closest to H)',
                     transform=ax_examples.transAxes, fontsize=9.5, va='center')
    ax_examples.text(.58, .58, ''.join(map(str, typical_sequence)),
                     transform=ax_examples.transAxes, fontsize=13, family='monospace',
                     weight='bold', color=COLORS['typical'], va='center')
    ax_examples.text(.02, .40,
                     f'Mode: P={probability_of(mode_sequence):.4g}; '
                     f'{membership_label(mode_is_typical, mode_is_most_likely)}',
                     transform=ax_examples.transAxes, fontsize=8.5, color='#526174', va='center')
    ax_examples.text(.02, .27,
                     f'Closest typical: {int(typical_sequence.sum())} ones, '
                     f'P={probability_of(typical_sequence):.4g}; '
                     f'{membership_label(typ_is_typical, typ_is_most_likely)}',
                     transform=ax_examples.transAxes, fontsize=8.5, color='#526174', va='center')

    # Classify the current Monte Carlo sample and group each dot by its 1-count.
    sampled = sample_state['values']
    counts = sampled.sum(axis=1)
    sample_codes = sampled @ binary_weights
    in_t = typical_mask[sample_codes]
    in_s = most_likely_mask[sample_codes]
    is_both = in_t & in_s
    is_t_only = in_t & ~in_s
    is_s_only = in_s & ~in_t
    is_neither = ~in_t & ~in_s

    ax_samples.clear()
    jitter_rng = np.random.default_rng(105)
    x_positions = counts + jitter_rng.uniform(-.22, .22, size=N_SAMPLES)
    y_positions = np.zeros(N_SAMPLES)
    for k in range(N + 1):
        indices = np.flatnonzero(counts == k)
        y_positions[indices] = np.arange(len(indices)) + .5

    ax_samples.scatter(x_positions[is_both], y_positions[is_both], s=18,
                       color=COLORS['both'], label='In both sets', zorder=4)
    ax_samples.scatter(x_positions[is_t_only], y_positions[is_t_only], s=18,
                       color=COLORS['typical'], label='Typical only', zorder=4)
    ax_samples.scatter(x_positions[is_s_only], y_positions[is_s_only], s=18,
                       color=COLORS['most_likely'], label='Most-likely only', zorder=4)
    ax_samples.scatter(x_positions[is_neither], y_positions[is_neither], s=18,
                       facecolors='none', edgecolors=COLORS['neither_edge'],
                       linewidths=1, label='Neither (unfilled)', zorder=3)
    ax_samples.set_xlim(-.6, N + .6)
    ax_samples.set_ylim(0, max(5, int(np.max(y_positions)) + 1))
    ax_samples.set_xticks(np.arange(N + 1))
    ax_samples.set_xlabel('Number of 1s in the sequence')
    ax_samples.set_ylabel('Draws, stacked within each count')
    ax_samples.set_title(f'{N_SAMPLES} sequences drawn independently from Bernoulli(p={p:.2f})',
                         fontsize=11, pad=9)
    ax_samples.grid(axis='x', alpha=.18)
    ax_samples.legend(loc='upper left', ncol=2, fontsize=8.5, frameon=True)

    t_count = int(np.sum(typical_mask))
    t_mass = float(np.sum(sequence_probabilities[typical_mask]))
    s_count = int(np.sum(most_likely_mask))
    s_mass = float(np.sum(sequence_probabilities[most_likely_mask]))
    overlap_count = int(np.sum(typical_mask & most_likely_mask))
    ax_examples.text(.02, .12,
                     rf'$H(p)={h:.3f}$ bits/symbol; $|T|={t_count}$, $P(T)={t_mass:.3f}$',
                     transform=ax_examples.transAxes, fontsize=8.2,
                     color=COLORS['ink'], va='center')
    ax_examples.text(.02, .035,
                     rf'$|S|={s_count}$, $P(S)={s_mass:.3f}$; overlap $={overlap_count}$ sequences',
                     transform=ax_examples.transAxes, fontsize=8.2,
                     color=COLORS['ink'], va='bottom')
    fig.canvas.draw_idle()


def resample(_):
    p = float(p_slider.val)
    sample_state['values'] = rng.binomial(1, p, size=(N_SAMPLES, N))
    update()


def change_p(_):
    resample(None)


p_slider.on_changed(change_p)
eps_t_slider.on_changed(update)
eps_s_slider.on_changed(update)
resample_button.on_clicked(resample)
update()
plt.show()
