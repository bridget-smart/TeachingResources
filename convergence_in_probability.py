"""Figure 1: almost sure convergence versus convergence in probability.
Run: python convergence_in_probability.py
Requires: numpy, matplotlib. Use an interactive matplotlib backend.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

rng = np.random.default_rng(2026)
MAX_N, PATHS = 1200, 48

# Dedicated columns for controls prevent text from colliding with plots.
def make_layout(title, options, active=0):
    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=False)
    fig.suptitle(title, fontsize=15, y=.975)
    ax_top = fig.add_axes([.32, .53, .64, .33])
    ax_bottom = fig.add_axes([.32, .18, .64, .26])
    ax_radio = fig.add_axes([.035, .56, .245, .25])
    radio = RadioButtons(ax_radio, options, active=active)
    ax_button = fig.add_axes([.055, .36, .19, .065])
    button = Button(ax_button, 'Resample paths')
    ax_slider = fig.add_axes([.40, .085, .47, .035])
    slider = Slider(ax_slider, 'Show first n', 20, MAX_N, valinit=250, valstep=10)
    fig.text(.035, .89, 'Choose example', fontsize=11, weight='bold')
    return fig, ax_top, ax_bottom, radio, button, slider

# FIGURE 1: convergence in probability versus almost surely.
opts1 = ['Almost surely', 'In probability only', 'Neither (diverges)']
f1, a1, b1, r1, btn1, s1 = make_layout('Almost sure convergence versus convergence in probability', opts1, 1)
state1 = {}

def sample1():
    n = np.arange(1, MAX_N+1)
    case = r1.value_selected
    p = 2.0**(-n) if case == opts1[0] else (1/n if case == opts1[1] else np.full(MAX_N,.5))
    state1['p'] = p
    state1['samples'] = rng.random((PATHS, MAX_N)) < p

def draw1(_=None):
    end = int(s1.val)
    p, arr = state1['p'], state1['samples']
    a1.clear(); b1.clear()
    a1.imshow(arr[:,:end], origin='lower', interpolation='nearest', aspect='auto',
              extent=[.5,end+.5,.5,PATHS+.5], cmap='Greys', vmin=0, vmax=1)
    a1.set_title('Sample paths: white = 0, black = 1', fontsize=11, pad=10)
    a1.set_ylabel('Path index')
    a1.set_xlim(.5,end+.5)
    x=np.arange(1,end+1)
    b1.plot(x,p[:end],label='Exact probability',linewidth=2.3)
    b1.plot(x,arr[:,:end].mean(axis=0),label='Fraction across 48 paths',alpha=.65,linewidth=1)
    b1.set_ylim(-.04,1.04); b1.set_xlim(1,end)
    b1.set_xlabel('n'); b1.set_ylabel(r'$P(|X_n|>1/2)$')
    b1.grid(alpha=.18); b1.legend(loc='upper right',fontsize=9)
    descriptions = {
        opts1[0]: r'$P(X_n=1)=2^{-n}$. Deviations occur only finitely often a.s.',
        opts1[1]: r'$P(X_n=1)=1/n$. Probability tends to zero; infinitely many deviations a.s.',
        opts1[2]: r'$P(X_n=1)=1/2$. No convergence to zero in probability or a.s.'}
    f1.texts[:] = [t for t in f1.texts if not getattr(t,'_dynamic_note',False)]
    t=f1.text(.32,.465,descriptions[r1.value_selected],fontsize=10)
    t._dynamic_note=True
    f1.canvas.draw_idle()

def change1(_): sample1();draw1()
r1.on_clicked(change1)
btn1.on_clicked(change1)
s1.on_changed(draw1)
sample1();draw1()


plt.show()
