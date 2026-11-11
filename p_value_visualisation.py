"""Interactive first look at a p-value using a fair-coin test.

Run: python p_value_visualisation.py
Requires: numpy and matplotlib.
"""
from math import comb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Patch


N_FLIPS = 20
rng = np.random.default_rng(2026)
outcomes = np.arange(N_FLIPS + 1)
null_probabilities = np.array([comb(N_FLIPS, k) for k in outcomes]) / 2**N_FLIPS

fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=False)
fig.suptitle('What does a p-value measure?', fontsize=16, y=.975)

# Controls and explanation sit in a quiet left column.
fig.text(.045, .89, 'The question', fontsize=11, weight='bold')
fig.text(.045, .84, 'Is this coin fair?', fontsize=13)
fig.text(.045, .79, r'Null model: $p(\mathrm{H})=1/2$' '\n'
         r'Assume 20 independent flips.' '\n\n'
         r'Test statistic: $K=$ number of heads' '\n'
         r'Under the null: $K\sim\mathrm{Binomial}(20,1/2)$',
         fontsize=10.5, va='top', linespacing=1.5)

fig.text(.045, .52, 'Set the observed result', fontsize=11, weight='bold')
ax_heads = fig.add_axes([.085, .46, .18, .035])
heads_slider = Slider(ax_heads, 'Heads', 0, N_FLIPS, valinit=10,
                      valstep=1, valfmt='%0.0f')
ax_button = fig.add_axes([.065, .35, .21, .06])
draw_button = Button(ax_button, 'Draw a fair-coin sample')

fig.text(.045, .25,
         'The p-value is the chance, if the\n'
         'coin is fair, of a result at least as\n'
         'far from 10 heads as the one observed.\n\n'
         'It is not the probability that the\n'
         'coin is fair.',
         fontsize=10, va='top', linespacing=1.35)

# Right column: the observed sample, then the distribution predicted by H₀.
ax_sample = fig.add_axes([.35, .72, .61, .16])
ax_null = fig.add_axes([.35, .30, .61, .32])
ax_result = fig.add_axes([.35, .07, .61, .17])
ax_result.set_facecolor('#eef4f8')
for spine in ax_result.spines.values():
    spine.set_color('#c7d5e2')
ax_result.set_xticks([])
ax_result.set_yticks([])
statistic_text = ax_result.text(.25, .70, '', ha='center', va='center', fontsize=12,
                                transform=ax_result.transAxes)
pvalue_text = ax_result.text(.75, .70, '', ha='center', va='center', fontsize=12,
                             transform=ax_result.transAxes)
tail_text = ax_result.text(.5, .35, '', ha='center', va='center', fontsize=9,
                           transform=ax_result.transAxes, color='#526174')
significance_text = ax_result.text(.5, .10, r'Significance level: $\alpha=0.05$; reject $H_0$ if $p\leq\alpha$',
                                   ha='center', va='center', fontsize=9,
                                   transform=ax_result.transAxes, color='#526174')

sample = rng.binomial(1, .5, size=N_FLIPS)
sample_cmap = ListedColormap(['#dce7f0', '#00A6FF'])


def make_sample_with_heads(heads):
    flips = np.zeros(N_FLIPS, dtype=int)
    if heads:
        flips[rng.choice(N_FLIPS, size=int(heads), replace=False)] = 1
    return flips


def draw():
    observed_heads = int(np.sum(sample))
    distance = abs(observed_heads - N_FLIPS / 2)
    as_extreme = np.abs(outcomes - N_FLIPS / 2) >= distance
    p_value = float(np.sum(null_probabilities[as_extreme]))

    ax_sample.clear()
    ax_sample.imshow(sample[None, :], origin='lower', aspect='auto',
                      extent=[-.5, N_FLIPS - .5, 0, 1], cmap=sample_cmap,
                      vmin=0, vmax=1, interpolation='nearest')
    for flip_index, result in enumerate(sample):
        ax_sample.text(flip_index, .5, 'H' if result else 'T',
                       ha='center', va='center', fontsize=9,
                       color='white' if result else '#25364a', weight='bold')
    ax_sample.set_xlim(-.5, N_FLIPS - .5)
    ax_sample.set_ylim(0, 1)
    ax_sample.set_yticks([])
    ax_sample.set_xticks(np.arange(0, N_FLIPS, 2))
    ax_sample.set_xticklabels(np.arange(1, N_FLIPS + 1, 2))
    ax_sample.set_xlabel('Flip number')
    ax_sample.set_title('The observed sample: H = head, T = tail', fontsize=11, pad=9)

    colors = np.where(as_extreme, '#00A6FF', '#d5dce5')
    ax_null.clear()
    bars = ax_null.bar(outcomes, null_probabilities, width=.82,
                       color=colors, edgecolor='white', linewidth=.6)
    bars[observed_heads].set_edgecolor('#14213d')
    bars[observed_heads].set_linewidth(2.2)
    ax_null.set_xlim(-.7, N_FLIPS + .7)
    ax_null.set_ylim(0, max(null_probabilities) * 1.32)
    ax_null.set_xticks(np.arange(0, N_FLIPS + 1, 2))
    ax_null.set_xlabel('Number of heads, K')
    ax_null.set_ylabel('Probability under the null')
    ax_null.set_title('Null distribution: results at least this extreme are blue',
                      fontsize=11, pad=9)
    ax_null.grid(axis='y', alpha=.2)
    ax_null.set_axisbelow(True)
    ax_null.legend(handles=[
        Patch(facecolor='#00A6FF', label='Counted in the p-value'),
        Patch(facecolor='#d5dce5', label='Not as extreme'),
    ], loc='upper left', fontsize=9, frameon=True)

    statistic_text.set_text(f'Test statistic: K = {observed_heads} heads')
    pvalue_text.set_text('p-value' + f'\n{p_value:.4g}')
    tail_text.set_text(r'$p=P_{H_0}(|K-10|\geq|K_{obs}-10|)$')
    fig.canvas.draw_idle()


def set_observed_heads(value):
    global sample
    sample = make_sample_with_heads(int(value))
    draw()


def draw_fair_sample(_):
    global sample
    sample = rng.binomial(1, .5, size=N_FLIPS)
    heads_slider.set_val(int(np.sum(sample)))
    draw()


heads_slider.on_changed(set_observed_heads)
draw_button.on_clicked(draw_fair_sample)
draw()
plt.show()
