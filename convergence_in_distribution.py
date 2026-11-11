"""Interactive illustration of convergence in distribution vs probability.
Run: python convergence_in_distribution.py
Requires: numpy and matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from math import erfc

MAX_N = 1200

# Let Z and W_n be independent standard normals and set
# X_n = sqrt(1 - n^(-2a)) Z + n^(-a) W_n.
# Every X_n has the same N(0, 1) density as Z. For a > 0, X_n -> Z in
# probability; for a = 0, X_n is independent of Z, so convergence is in
# distribution only.
fig = plt.figure(figsize=(13.5, 10.0), constrained_layout=False)
fig.suptitle('Same density, different relationship to the limit', fontsize=15, y=.975)
ax_pdf = fig.add_axes([.37, .72, .59, .17])
ax_paths = fig.add_axes([.37, .43, .59, .17])
ax_prob = fig.add_axes([.37, .13, .59, .18])
fig.text(.035, .89, 'Tune the relationship', fontsize=11, weight='bold')
ax_alpha = fig.add_axes([.09, .81, .17, .025])
alpha_slider = Slider(ax_alpha, r'Decay $a$', 0, 1, valinit=1, valstep=.05, dragging=True)
alpha_slider.drawon = False
alpha_slider.valtext.set_x(1.08)
fig.text(.055, .70,
         r'$X_n=\sqrt{1-n^{-2a}}\,Z+n^{-a}W_n$' '\n\n'
         r'$Z,W_n\sim N(0,1)$ independently' '\n\n'
         r'Each $X_n$ combines $Z$ with zero-centred' '\n'
         r'noise $n^{-a}W_n$; variance $n^{-2a}$' '\n'
         r'decays faster for larger $a$ (for $n>1$).' '\n\n'
         r'$a>0$: almost surely, hence also' '\n'
         r'in probability and in distribution' '\n'
         r'$a=0$: in distribution, not probability',
         fontsize=10, va='top', linespacing=1.35)

state = {'seed': 2026}
path_image = None
prob_line = None
note_artist = None
x_pdf = np.linspace(-4, 4, 501)
pdf = np.exp(-x_pdf**2 / 2) / np.sqrt(2 * np.pi)
epsilon = 1.0


def draw_all():
    global path_image, prob_line, note_artist
    alpha = float(alpha_slider.val)
    ns = np.arange(1, MAX_N + 1, dtype=float)
    sigma = ns ** (-alpha)
    variance = 2 * (1 - np.sqrt(np.maximum(0, 1 - sigma**2)))
    probability = np.array([
        erfc(epsilon / np.sqrt(2 * v)) if v > 1e-14 else 0.0
        for v in variance
    ])

    ax_pdf.clear()
    ax_pdf.plot(x_pdf, pdf, color='black', linewidth=2.4, label=r'Target $Z$')
    ax_pdf.plot(x_pdf, pdf, color='#00A6FF', linewidth=2.0, linestyle='--',
                label=r'$X_n$ (same density for every $n$)')
    ax_pdf.set_title(r'Probability densities: $X_n \sim Z \sim N(0,1)$', fontsize=11, pad=10)
    ax_pdf.set_xlabel('value')
    ax_pdf.set_ylabel('density')
    ax_pdf.set_xlim(-4, 4)
    ax_pdf.set_ylim(0, .44)
    ax_pdf.grid(alpha=.18)
    ax_pdf.legend(loc='upper right', fontsize=9)

    path_rng = np.random.default_rng(state['seed'])
    path_count, path_n = 18, min(500, MAX_N)
    z_paths = path_rng.standard_normal((path_count, 1))
    w_paths = path_rng.standard_normal((path_count, path_n))
    path_ns = np.arange(1, path_n + 1, dtype=float)
    path_sigma = path_ns ** (-alpha)
    path_values = (np.sqrt(np.maximum(0, 1 - path_sigma**2))[None, :] * z_paths
                   + path_sigma[None, :] * w_paths)
    if path_image is None:
        path_image = ax_paths.imshow(
            path_values, origin='lower', interpolation='nearest', aspect='auto',
            extent=[.5, path_n + .5, .5, path_count + .5], cmap='coolwarm',
            vmin=-3, vmax=3)
        fig.colorbar(path_image, ax=ax_paths, fraction=.025, pad=.01, label=r'$X_n$ value')
        ax_paths.set_title(r'Sample paths: each row uses one fixed $Z$', fontsize=11, pad=8)
        ax_paths.set_ylabel('Path index')
        ax_paths.set_xlim(.5, path_n + .5)
        ax_paths.set_xlabel('n')
    else:
        path_image.set_data(path_values)

    if prob_line is None:
        (prob_line,) = ax_prob.plot(
            ns, probability, color='tab:blue', linewidth=2.3,
            label=r'Exact $P(|X_n-Z|>1)$')
        ax_prob.set_title(r'How often are $X_n$ and $Z$ more than 1 apart?', fontsize=11, pad=10)
        ax_prob.set_xlabel('n')
        ax_prob.set_ylabel('probability')
        ax_prob.set_xlim(1, MAX_N)
        ax_prob.set_ylim(-.03, 1.03)
        ax_prob.grid(alpha=.18)
        ax_prob.legend(loc='upper right', fontsize=9)
    else:
        prob_line.set_ydata(probability)

    if alpha == 0:
        note = (r'$a=0$: $X_n$ is independent of $Z$. The densities match exactly, '
                r'but the chance of being far apart stays positive: distribution only.')
    else:
        note = (rf'$a={alpha:g}>0$: added noise shrinks with $n$; '
                r'$X_n$ converges to $Z$ almost surely, hence in probability and distribution.')
    if note_artist is None:
        note_artist = fig.text(.37, .365, note, fontsize=10)
    else:
        note_artist.set_text(note)
    fig.canvas.draw_idle()


def update_during_drag(value):
    """Update only the probability curve and note while the slider moves."""
    alpha = float(value)
    ns = np.arange(1, MAX_N + 1, dtype=float)
    sigma = ns ** (-alpha)
    variance = 2 * (1 - np.sqrt(np.maximum(0, 1 - sigma**2)))
    probability = np.array([
        erfc(epsilon / np.sqrt(2 * v)) if v > 1e-14 else 0.0
        for v in variance
    ])
    prob_line.set_ydata(probability)
    if alpha == 0:
        note_artist.set_text(
            r'$a=0$: $X_n$ is independent of $Z$. The densities match exactly, '
            r'but the chance of being far apart stays positive: distribution only.')
    else:
        note_artist.set_text(
            rf'$a={alpha:g}>0$: added noise shrinks with $n$; '
            r'$X_n$ converges to $Z$ almost surely, hence in probability and distribution.')
    fig.canvas.draw_idle()


def finish_drag(event):
    if event.inaxes == ax_alpha:
        draw_all()


def resample_paths(_):
    state['seed'] += 1
    draw_all()


alpha_slider.on_changed(update_during_drag)
fig.canvas.mpl_connect('button_release_event', finish_drag)
ax_button = fig.add_axes([.055, .43, .19, .055])
button = Button(ax_button, 'Resample paths')
button.on_clicked(resample_paths)
draw_all()
plt.show()
