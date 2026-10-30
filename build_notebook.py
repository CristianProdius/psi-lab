"""Generează lab1.ipynb (Lucrarea de laborator N1, PASI) din celulele de mai jos.

Rulare:  uv run python build_notebook.py && uv run jupyter nbconvert --execute --inplace --to notebook lab1.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Lucrarea de laborator N 1 — Modelarea semnalelor elementare

**Disciplina:** Proiectarea şi studierea semnalelor (PASI), Masterat 2024
**Universitatea Tehnică a Moldovei**, Departamentul Microelectronică şi Inginerie Biomedicală

**Scopul lucrării:** modelarea semnalelor elementare (dreptunghiular, sinusoidal, exponenţial, impuls Gauss,
sinc, chirp, Dirichlet) şi studiul eşantionării semnalelor continue (teorema eşantionării, fenomenul de aliasing).

**Implementare:** Python 3 — NumPy, SciPy (`scipy.signal`, `scipy.special`), Matplotlib — în locul MATLAB.
""")

md(r"""
## 1. Consideraţii teoretice — eşantionarea semnalelor continue

Fie un semnal sinusoidal continuu

$$x_a(t) = A\cos(\Omega t + \varphi), \qquad \Omega = 2\pi F,\quad -\infty < t < \infty .$$

Eşantionarea uniformă cu perioada $T = 1/F_s$ ($F_s$ — frecvenţa de eşantionare) înseamnă $t = nT$:

$$x(n) \equiv x_a(nT) = A\cos(\Omega n T + \varphi) = A\cos\!\left(\frac{2\pi n F}{F_s} + \varphi\right).$$

Se obţine deci semnalul în timp discret

$$x(n) = A\cos(\omega n + \varphi) = A\cos(2\pi f n + \varphi),$$

unde $\omega = \Omega T$ şi $f = F/F_s$ (frecvenţa normalizată, cicli/eşantion).

Deoarece sinusoidele discrete sunt unice doar pentru $-\pi \le \omega \le \pi$, adică $-\tfrac12 \le f \le \tfrac12$, rezultă:

$$-\frac{F_s}{2} \le F \le \frac{F_s}{2}.$$

Orice frecvenţă $F$ din afara acestui interval apare, după eşantionare, ca frecvenţa
$F' = F - kF_s$ din interiorul lui (*aliasing*). Graficul de mai jos (analogul figurii 1.1) arată această
corespondenţă: frecvenţa analogică $F$ se „pliază” în intervalul $[-F_s/2,\, F_s/2]$.
""")

md("## 2. Configurare şi funcţii auxiliare")

code(r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, special

%matplotlib inline
%config InlineBackend.figure_format = "retina"

# Echivalentul lui set(gca,'FontName','Arial Cyr','FontSize',16) din îndrumar, aplicat global o singură dată.
plt.rcParams.update({
    "figure.figsize": (10, 4),
    "font.family": "DejaVu Sans",  # suportă diacriticele româneşti
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.grid": True,
    "grid.alpha": 0.4,
})
""")

code(r'''
def show(t, y, title, xlabel="timpul(s)", ylabel="functia y(t)", kind="plot", ax=None):
    """Echivalentul secvenţei MATLAB plot/stem + grid + title + xlabel + ylabel."""
    if ax is None:
        _, ax = plt.subplots()
    if kind == "stem":
        ax.stem(t, y, basefmt="k-", markerfmt="o")
    else:
        ax.plot(t, y)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return ax


def apparent_freq(F, Fs):
    """Frecvenţa (cu semn) la care apare F după eşantionare cu Fs, în intervalul [-Fs/2, Fs/2)."""
    return (np.asarray(F) + Fs / 2) % Fs - Fs / 2


def amplitude_spectrum(x, Fs):
    """Spectrul de amplitudine unilateral (amplitudinea în V pentru o sinusoidă pe un număr întreg de perioade)."""
    N = len(x)
    X = np.fft.rfft(x)
    amp = np.abs(X) / N
    amp[1:-1] *= 2  # componentele pozitive preiau energia celor negative (exceptând DC şi Nyquist)
    return np.fft.rfftfreq(N, d=1 / Fs), amp
''')

code(r"""
# Figura 1.1 — corespondenţa dintre frecvenţa analogică F şi cea aparentă după eşantionare (Fs = 256 Hz)
Fs = 256
F = np.linspace(0, 600, 3001)
Fa = apparent_freq(F, Fs)
Fa[np.abs(np.diff(Fa, prepend=Fa[0])) > Fs / 2] = np.nan  # rupe linia la salturile de pliere

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(F, Fa, lw=2)
ax.axhspan(-Fs / 2, Fs / 2, color="tab:green", alpha=0.06)
for f_true in (100, 156, 356):
    fa = apparent_freq(f_true, Fs)
    ax.plot(f_true, fa, "o", ms=8)
    ax.annotate(f"{f_true} Hz → {fa:+.0f} Hz", (f_true, fa), textcoords="offset points", xytext=(8, 8))
for k in range(1, 3):
    ax.axvline(k * Fs, color="gray", ls=":")
ax.set(title="Plierea frecvenţei la eşantionare, Fs = 256 Hz",
       xlabel="Frecvenţa analogică F (Hz)", ylabel="Frecvenţa aparentă F' (Hz)",
       yticks=[-Fs / 2, -100, 0, 100, Fs / 2])
plt.show()
""")

md("""
## 3. Desfăşurarea lucrării — modelarea semnalelor elementare

Fiecare punct reproduce programul MATLAB din îndrumar (indicat deasupra codului) cu echivalentul Python.
Toate echivalenţele (`square`, `gausspulse`, `chirp`, `sinc`, `diric`) au fost verificate numeric faţă de
definiţiile MATLAB.
""")

md(r"""
### 1.1. Semnal periodic dreptunghiular — `square`

```matlab
A=1; w0=10*pi; rho=0.5; t=0:.001:1;
sq=A*square(w0*t+rho);
```
`scipy.signal.square` are aceeaşi convenţie ca MATLAB: perioadă $2\pi$, $+1$ pe $[0,\pi)$ şi $-1$ pe $[\pi, 2\pi)$.
Cu $\omega_0 = 10\pi$ rad/s se obţine $F_0 = 5$ Hz (5 perioade în 1 s). Observaţie: `rho` este adunat
**argumentului**, deci reprezintă o defazare de 0.5 rad, nu factorul de umplere (acesta ar fi al doilea
argument al lui `square`).
""")
code(r"""
A, w0, rho = 1, 10 * np.pi, 0.5
t = np.arange(0, 1 + 1e-9, 0.001)  # echivalentul 0:.001:1 (1001 eşantioane)
sq = A * signal.square(w0 * t + rho)

ax = show(t, sq, "Semnal periodic in forma dreptunghiulara", xlabel="t,sec", ylabel="X(t)")
ax.set_ylim(-1.3, 1.3)
plt.show()
""")

md(r"""
### 1.2. Semnal periodic sinusoidal — `cos`

```matlab
A=4; w0=20*pi; phi=pi/6; t=0:.001:1;
cosine=A*cos(w0*t+phi);
```
$F_0 = \omega_0 / 2\pi = 10$ Hz, amplitudine 4, fază iniţială $\pi/6$.
""")
code(r"""
A, w0, phi = 4, 20 * np.pi, np.pi / 6
t = np.arange(0, 1 + 1e-9, 0.001)
cosine = A * np.cos(w0 * t + phi)

show(t, cosine, "Exemplu de utilizare a procedurii cos")
plt.show()
""")

md(r"""
### 1.3. Semnal discret sinusoidal — `cos` + `stem`

```matlab
A=4; w0=20*pi; phi=pi/6; t=0:.005:1;
cosine=A*cos(w0*t+phi);  stem(t,cosine)
```
Pas de 0.005 s, adică $F_s = 200$ Hz, deci $F_s / F_0 = 20$ eşantioane pe perioadă. Frecvenţa normalizată
$f = F_0/F_s = 0.05 \le 0.5$, deci eşantionarea respectă teorema lui Kotelnikov–Shannon. Al doilea grafic
mărește prima perioadă şi suprapune semnalul continuu.
""")
code(r"""
t_d = np.arange(0, 1 + 1e-9, 0.005)
cosine_d = A * np.cos(w0 * t_d + phi)

show(t_d, cosine_d, "Exemplu de utilizare a procedurii cos", kind="stem")
plt.show()

# Detaliu: prima perioadă, semnal continuu vs. eşantionat
fig, ax = plt.subplots()
mask_c, mask_d = t <= 0.1, t_d <= 0.1
ax.plot(t[mask_c], cosine[mask_c], "--", color="gray", label="x(t), continuu")
ax.stem(t_d[mask_d], cosine_d[mask_d], basefmt="k-", label="x(nT), Fs = 200 Hz")
ax.set(title="Semnal continuu şi eşantionat (prima perioadă)", xlabel="timpul(s)", ylabel="functia y(t)")
ax.legend(loc="lower right")
plt.show()
""")

md(r"""
### 1.4. Semnal exponenţial crescător — `exp`

```matlab
B=1; a=5; t=0:.001:1;
x=B*exp(a*t);
```
Pentru $a > 0$ semnalul creşte nelimitat; la $t = 1$ s valoarea este $e^5 \approx 148.4$.
""")
code(r"""
B, a = 1, 5
t = np.arange(0, 1 + 1e-9, 0.001)
x = B * np.exp(a * t)

show(t, x, "Exemplu de utilizare a procedurii exp")
plt.show()
print(f"x(1) = {x[-1]:.2f}  (e^5 = {np.exp(5):.2f})")
""")

md(r"""
### 1.5. Sinusoidă modulată gaussian — `gauspuls`

```matlab
t=-10:.01:10;
y=0.75*gauspuls(t+3,1,0.5);
```
`gauspuls(t, fc, bw)` corespunde lui `scipy.signal.gausspulse(t, fc=fc, bw=bw)` (lăţimea de bandă relativă
măsurată la −6 dB în ambele implementări):

$$y(t) = 0.75\, e^{-a (t+3)^2} \cos\big(2\pi f_c (t+3)\big),\qquad f_c = 1\ \text{Hz},\ bw = 0.5.$$

Deplasarea `t+3` centrează impulsul în $t = -3$ s.
""")
code(r"""
t = np.arange(-10, 10 + 1e-9, 0.01)
y = 0.75 * signal.gausspulse(t + 3, fc=1, bw=0.5)

ax = show(t, y, "Exemplu de utilizare a procedurii gauspuls")
ax.axvline(-3, color="gray", ls=":", lw=1)
plt.show()
""")

md(r"""
### 1.6. Funcţia sinc — transformata Fourier a unui impuls dreptunghiular

```matlab
t=0:.01:50;
y1=0.7*sinc(pi*(t-25)/5);
```
MATLAB şi NumPy folosesc ambele sinc-ul **normalizat**, $\operatorname{sinc}(x) = \sin(\pi x)/(\pi x)$.
Maximul (0.7) este la $t = 25$ s, iar zerourile apar la $t - 25 = 5k/\pi$, $k \ne 0$
(adică la intervale de $\approx 1.59$ s).
""")
code(r"""
t = np.arange(0, 50 + 1e-9, 0.01)
y1 = 0.7 * np.sinc(np.pi * (t - 25) / 5)

ax = show(t, y1, "Exemplu de utilizare a procedurii sinc")
zeros = 25 + 5 * np.array([k for k in range(-5, 6) if k]) / np.pi
ax.plot(zeros, np.zeros_like(zeros), "x", color="tab:red", label="zerouri teoretice 25 ± 5k/π")
ax.legend(loc="upper right")
plt.show()
""")

md(r"""
### 1.7. Semnal chirp (frecvenţă liniar variabilă) — `chirp`

```matlab
t=0:0.001:1;
y=0.75*chirp(t);
```
`chirp(t)` din MATLAB foloseşte implicit $f_0 = 0$ Hz, $t_1 = 1$ s, $f_1 = 100$ Hz, variaţie liniară. Aceşti
parametri trebuie daţi explicit în `scipy.signal.chirp`. Frecvenţa instantanee: $f(t) = f_0 + (f_1 - f_0)\,t/t_1$.
Spectrograma confirmă creşterea liniară a frecvenţei.
""")
code(r"""
t = np.arange(0, 1 + 1e-9, 0.001)
y = 0.75 * signal.chirp(t, f0=0, t1=1, f1=100, method="linear")

show(t, y, "Exemplu de utilizare a procedurii chirp")
plt.show()

# Verificare în domeniul timp–frecvenţă (Fs = 1000 Hz)
fig, ax = plt.subplots()
f_s, t_s, Sxx = signal.spectrogram(y, fs=1000, nperseg=64, noverlap=56)
S_db = 10 * np.log10(Sxx + 1e-12)
ax.pcolormesh(t_s, f_s, S_db, shading="gouraud", cmap="viridis", vmin=S_db.max() - 50)
ax.plot(t, 100 * t, "r--", lw=2, label="f(t) = 100·t (teoretic)")
ax.set(title="Spectrograma semnalului chirp", xlabel="timpul(s)", ylabel="frecventa (Hz)", ylim=(0, 200))
ax.legend(loc="upper left")
ax.grid(False)
plt.show()
""")

md(r"""
### 1.8. Funcţia Dirichlet — `diric`

```matlab
t=0:.01:50;
y=0.7*diric(t,4);
```
$$D_n(x) = \frac{\sin(n x/2)}{n \sin(x/2)},\qquad n = 4.$$

Este periodică cu perioada $2\pi$ pentru $n$ par ($4\pi$ pentru $n$ impar). În punctele $x = 2\pi k$ ia valoarea
limită $(-1)^{k(n-1)}$, deci pentru $n = 4$ alternează între $+1$ şi $-1$ (×0.7). `scipy.special.diric` are
aceeaşi definiţie ca MATLAB.
""")
code(r"""
t = np.arange(0, 50 + 1e-9, 0.01)
y = 0.7 * special.diric(t, 4)

ax = show(t, y, "Exemplu de utilizare a procedurii diric")
k = np.arange(0, int(50 // (2 * np.pi)) + 1)
ax.plot(2 * np.pi * k, 0.7 * (-1.0) ** (k * 3), "o", color="tab:red", label=r"$0.7\cdot(-1)^{k(n-1)}$ la $x = 2\pi k$")
ax.set_ylim(-0.8, 1.05)
ax.legend(loc="upper right")
plt.show()
""")

md(r"""
## 4. Teorema eşantionării

### 1.8 (bis). Suma a două sinusoide: 100 Hz + 156 Hz, Fs = 256 Hz

În îndrumar acest punct este numerotat tot 1.8.

```matlab
t=(1:512); f1=100; f2=156; fe=256;
y=sin(2*pi*f1/fe*t)+sin(2*pi*f2/fe*t);
plot(t/fe,y)
```
Frecvenţa Nyquist este $F_s/2 = 128$ Hz, iar $f_2 = 156$ Hz o **depăşeşte**. Pentru $n$ întreg:

$$\sin\!\left(2\pi \tfrac{156}{256} n\right) = \sin\!\left(2\pi n - 2\pi \tfrac{100}{256} n\right) = -\sin\!\left(2\pi \tfrac{100}{256} n\right).$$

Sinusoida de 156 Hz devine, după eşantionare, o sinusoidă de 100 Hz **în antifază**, deci suma este identic
nulă. Ce rămâne în grafic sunt doar erori de rotunjire de ordinul $10^{-13}$.
""")
code(r"""
n = np.arange(1, 513)          # t=(1:512) -> 2 s la fe = 256 Hz
f1, f2, fe = 100, 156, 256
s1 = np.sin(2 * np.pi * f1 / fe * n)
s2 = np.sin(2 * np.pi * f2 / fe * n)
y = s1 + s2

show(n / fe, y, "Suma sinusoidelor 100 Hz + 156 Hz (fe = 256 Hz), scara automata",
     xlabel="timp (s)", ylabel="amplitudine (V)")
plt.show()

print(f"max |y|                    = {np.max(np.abs(y)):.3e} V   (zgomot de rotunjire)")
print(f"frecventa aparenta a 156 Hz = {apparent_freq(f2, fe):+.0f} Hz")
assert np.allclose(y, 0, atol=1e-9), "suma ar trebui sa fie identic nula"
assert np.allclose(s2, -s1, atol=1e-9), "156 Hz trebuie sa fie aliasul inversat al lui 100 Hz"
""")

code(r"""
# Aceeaşi sumă la scară fizică (±2 V) şi componentele ei: s2 este oglinda exactă a lui s1
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
w = n <= 20
axes[0].stem(n[w] / fe, s1[w], linefmt="C0-", markerfmt="C0o", basefmt="k-", label="sin 100 Hz")
axes[0].stem(n[w] / fe, s2[w], linefmt="C1--", markerfmt="C1s", basefmt="k-", label="sin 156 Hz")
axes[0].set(title="Componentele esantionate (primele 20 de esantioane)", xlabel="timp (s)", ylabel="amplitudine (V)")
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
axes[1].plot(n / fe, y)
axes[1].set(title="Suma la scara fizica", xlabel="timp (s)", ylabel="amplitudine (V)", ylim=(-2.2, 2.2))
plt.tight_layout()
plt.show()

# Spectrele: ambele componente cad pe acelaşi bin de 100 Hz, iar suma nu mai are nicio componentă
fig, ax = plt.subplots()
for x_, lbl, st in ((s1, "sin 100 Hz", "-"), (s2, "sin 156 Hz", "--"), (y, "suma", "-")):
    f_ax, amp = amplitude_spectrum(x_, fe)
    ax.plot(f_ax, amp, st, label=lbl, lw=2 if lbl == "suma" else 1.5)
ax.axvline(fe / 2, color="gray", ls=":", label="Fs/2 = 128 Hz")
ax.set(title="Spectrul de amplitudine (FFT, N = 512, rezolutie 0.5 Hz)",
       xlabel="frecventa (Hz)", ylabel="amplitudine (V)", xlim=(0, 128))
ax.legend()
plt.show()
""")

md(r"""
### 1.9. Compararea sinusoidelor de 100 Hz şi 356 Hz, Fs = 256 Hz

```matlab
t=(1:100); f1=100; f2=356; fe=256;
y1=sin(2*pi*f1/fe*t); y2=sin(2*pi*f2/fe*t);
subplot(211); plot(t/fe,y1); ...  subplot(212); plot(t/fe,y2); ...
stem(t,y1)
stem(t,y2)
```
$356 = 100 + 256$, deci $\sin(2\pi\tfrac{356}{256}n) = \sin(2\pi\tfrac{100}{256}n + 2\pi n) = \sin(2\pi\tfrac{100}{256}n)$.
Cele două secvenţe sunt **identice**: după eşantionare nu se mai poate distinge între 100 Hz şi 356 Hz.
""")
code(r"""
n = np.arange(1, 101)
f1, f2, fe = 100, 356, 256
y1 = np.sin(2 * np.pi * f1 / fe * n)
y2 = np.sin(2 * np.pi * f2 / fe * n)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
show(n / fe, y1, "sinusoida de frecventa 100 Hz", xlabel="", ylabel="amplitudine (V)", ax=ax1)
show(n / fe, y2, "sinusoida de frecventa 356 Hz", xlabel="timp (s)", ylabel="amplitudine (V)", ax=ax2)
plt.tight_layout()
plt.show()

print(f"max |y1 - y2|               = {np.max(np.abs(y1 - y2)):.3e}")
print(f"frecventa aparenta a 356 Hz = {apparent_freq(f2, fe):+.0f} Hz")
assert np.allclose(y1, y2, atol=1e-9), "secventele trebuie sa fie identice (aliasing)"
""")

code(r"""
# stem(t,y1) şi stem(t,y2): reprezentarea în funcţie de indicele eşantionului n
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
show(n, y1, "stem(t, y1): 100 Hz", xlabel="", ylabel="amplitudine (V)", kind="stem", ax=ax1)
show(n, y2, "stem(t, y2): 356 Hz", xlabel="n (indicele esantionului)", ylabel="amplitudine (V)",
     kind="stem", ax=ax2)
plt.tight_layout()
plt.show()
""")

code(r"""
# De ce sunt identice: ambele semnale continue trec exact prin aceleaşi eşantioane
n_zoom = np.arange(1, 9)
t_fine = np.linspace(n_zoom[0] / fe, n_zoom[-1] / fe, 4000)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(t_fine, np.sin(2 * np.pi * f2 * t_fine), color="tab:orange", lw=1, label="x(t) = sin(2π·356·t)")
ax.plot(t_fine, np.sin(2 * np.pi * f1 * t_fine), color="tab:blue", lw=2, label="x(t) = sin(2π·100·t)")
ax.plot(n_zoom / fe, np.sin(2 * np.pi * f1 / fe * n_zoom), "ko", ms=8, label="esantioane comune, Fs = 256 Hz")
ax.set(title="Aliasing: 356 Hz si 100 Hz produc aceleasi esantioane", xlabel="timp (s)", ylabel="amplitudine (V)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)
plt.tight_layout()
plt.show()
""")

md(r"""
## 5. Concluzii

1. Funcţiile MATLAB din îndrumar au echivalente directe în Python: `square`, `sawtooth`, `chirp`, `gausspulse`
   în `scipy.signal`, `diric` în `scipy.special`, iar `sinc`, `cos`, `exp` în NumPy. Echivalenţa a fost verificată
   numeric, inclusiv parametrii impliciţi care diferă (`chirp`: $f_1 = 100$ Hz; `gausspulse`: $f_c$ explicit).
2. Semnalele elementare modelate (dreptunghiular 5 Hz, cosinusoidal 10 Hz, exponenţial, impuls Gauss, sinc,
   chirp 0–100 Hz, Dirichlet) au formele şi parametrii prevăzuţi de teorie: zerourile sinc-ului, valorile
   limită ale funcţiei Dirichlet şi frecvenţa instantanee a chirp-ului (confirmată prin spectrogramă).
3. Un semnal eşantionat cu $F_s$ reprezintă corect doar frecvenţele $|F| \le F_s/2$. La $F_s = 256$ Hz
   (Nyquist 128 Hz):
   - **156 Hz** apare ca **−100 Hz**, adică o sinusoidă de 100 Hz în antifază, iar suma cu sinusoida de 100 Hz
     este identic nulă (punctul 1.8 bis);
   - **356 Hz** apare ca **+100 Hz**, iar secvenţa rezultată este identică cu cea a sinusoidei de 100 Hz
     (punctul 1.9).
4. Aliasing-ul este ireversibil: din eşantioane nu se mai poate determina frecvenţa originală. De aceea, înainte
   de eşantionare se aplică un filtru anti-aliasing (trece-jos cu $F_c < F_s/2$) sau se alege
   $F_s > 2F_{max}$ (teorema Kotelnikov–Shannon).
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
nbf.write(nb, "lab1.ipynb")
print(f"lab1.ipynb: {len(cells)} celule")
