"""
Terrain-aided navigation by radar-altimeter measurements with a
regularized particle filter (RPF, Epanechnikov kernel).
Scenario data: TP4 2026 ENSTA (300 km/h, heading 45°, 60 km, z0 = 8000 m).

Usage: python rpf_altimetrie.py [carte.mat] [seed]
In Spyder: Run (F5); the 2 figures appear in the Plots pane (no file export).
"""
import sys
import os
import numpy as np
from math import gamma, pi, sqrt
from scipy.io import loadmat

# Spyder / IPython: force inline backend so that figures stay docked in the
# Plots pane (no floating windows), whatever the Spyder graphics preference.
try:
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
except NameError:
    pass   # plain Python: default backend

import matplotlib.pyplot as plt

# MODIFICATION TP4 : fonctions complémentaires (boucle, MAP et BCR).
# Le filtre lui-même reste dans main ci-dessous.
from outils_tp import loop_command, kde_map, psd_sqrt

# ---------------------------------------------------------------------------
# Terrain model
# ---------------------------------------------------------------------------
class Terrain:
    def __init__(self, matfile):
        m = loadmat(matfile)
        self.h_MNT = np.asarray(m["h_MNT"], dtype=float)
        self.pasx = float(m["pasx_reel"].squeeze())
        self.pasy = float(m["pasy_reel"].squeeze())
        self.x_MNT = np.asarray(m["x_MNT"], dtype=float).ravel()
        self.y_MNT = np.asarray(m["y_MNT"], dtype=float).ravel()
        self.nrow = self.h_MNT.shape[0]
        # column-major flat view to reproduce MATLAB linear indexing exactly
        self.hF = self.h_MNT.ravel(order="F")

    def extent_km(self):
        return [self.x_MNT[0] * self.pasx / 1000, self.x_MNT[-1] * self.pasx / 1000,
                self.y_MNT[-1] * self.pasy / 1000, self.y_MNT[0] * self.pasy / 1000]

    def hobs(self, X):
        """Bilinear interpolation of terrain height at (X[0], X[1]).
        x = column index, y = row index of h_MNT (image convention)."""
        ncol = self.h_MNT.shape[1]
        # clamp to the map (particles leaving the DEM would index out of bounds)
        lam = np.clip(X[0] / self.pasx, 0.0, ncol - 1 - 1e-9)
        ix = np.floor(lam)
        lam = lam - ix
        mu = np.clip(1.0 + X[1] / self.pasy, 1.0, self.nrow - 1e-9)
        iy = np.floor(mu)
        mu = mu - iy
        k = (self.nrow * ix + iy).astype(np.int64) - 1   # MATLAB 1-based -> 0-based
        n = self.nrow
        hF = self.hF
        return ((1 - lam) * (1 - mu) * hF[k] + lam * (1 - mu) * hF[k + n]
                + (1 - lam) * mu * hF[k + 1] + lam * mu * hF[k + n + 1])


# ---------------------------------------------------------------------------
# Epanechnikov sampler (Deheuvels; Devroye p. 236)
# density ∝ (1 - ||x||^2) on the unit ball of R^d
# ---------------------------------------------------------------------------
def gene_epanech(d, n, rng):
    # ||X||^2 ~ Beta(d/2, 2). The MATLAB code draws it with Cheng's rejection
    # algorithm; numpy's Beta sampler gives the same distribution.
    beta = rng.beta(d / 2.0, 2.0, size=n)
    g = rng.standard_normal((d, n))
    return g * (np.sqrt(beta) / np.linalg.norm(g, axis=0))


def select(p, rng):
    """Multinomial resampling."""
    c = np.cumsum(p)
    c[-1] = 1.0
    return np.searchsorted(c, rng.random(p.size))


# ---------------------------------------------------------------------------
# MODIFICATION TP4 : les 6 arguments d'origine sont conservés, avec leurs
# valeurs par défaut. Les paramètres supplémentaires pilotent les expériences.
def main(matfile, seed=None, ChoixNoyau=2, coefR=1.5, plot=True, verbose=True,
         NbPart=5000, mu=0.3, CoefResampling=0.5, LoiInitiale="gauss",
         TypeFiltre="RPF", PositionInitiale=(150000., 150000.),
         Trajectoire="droite", BruitDynamique=1., Donnees=None,
         RetourDetails=False, PasMAP=0):
    rng = np.random.default_rng(seed)
    # MODIFICATION TP4 : flux séparés pour comparer les mêmes données avec
    # différents filtres ; changer de noyau ne change pas le bruit des mesures.
    rng_init, rng_dyn, rng_resamp, rng_reg = [np.random.default_rng(s)
        for s in np.random.SeedSequence(seed).spawn(4)]
    if TypeFiltre not in ("SIR", "RPF") or LoiInitiale not in ("gauss", "uniforme"):
        raise ValueError("TypeFiltre ou LoiInitiale inconnu")
    if ChoixNoyau not in (1, 2) or NbPart < 2 or not 0 < CoefResampling <= 1:
        raise ValueError("Paramètres invalides")
    terr = Terrain(matfile)
    d = 6

    # ======================= Données du scénario (TP4) =======================
    VitAvion = 300 * 1000 / 3600           # vitesse avion (m/s)
    CapAvion = 45 * pi / 180               # cap avion (rad)
    LongTraject = 60 * 1000                # longueur du vol (m)
    EtatVraiInitial = np.array([150 * 1000, 150 * 1000, 8000,
                                VitAvion * np.cos(CapAvion),
                                VitAvion * np.sin(CapAvion), 0.0])
    # MODIFICATION TP4 : changer de région sans toucher aux autres données.
    EtatVraiInitial[:2] = PositionInitiale
    dt = 1.0                               # période de mesure (s)
    SigmaBruit = 30.0                      # écart-type bruit de mesure (m)
    # Incertitude initiale : σxy = 3000 m, σz = 500 m, σv = 5 m/s  (donne P0)
    P0 = np.diag(np.array([3000, 3000, 500, 5, 5, 5], dtype=float) ** 2)
    RacineP0 = np.linalg.cholesky(P0)
    # MODIFICATION TP4 : 720 transitions et l'état initial, soit 721 états.
    T = int(round(LongTraject / (VitAvion * dt))) + 1

    # ===================== Filtre particulaire régularisé ====================
    # MODIFICATION TP4 : NbPart est désormais un argument (500 à 5000).
    Np = NbPart
    # ChoixNoyau : 1 = Epanechnikov, 2 = gaussien (TP4 : gaussien)
    # MODIFICATION TP4 : seuil et mu sont paramétrables ; ne pas les confondre.
    Nth = CoefResampling * Np
    # ==========================================================================

    F = np.eye(d)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    Q = np.diag(np.array([.5, .5, .1, .1, .1, .01]) ** 2)
    LQ = np.linalg.cholesky(Q)

    # MODIFICATION TP4 : commande de virage connue, nulle pour le MRU.
    Commande = loop_command(T-1) if Trajectoire == "boucle" else np.zeros((d,T))
    if Trajectoire not in ("droite", "boucle"):
        raise ValueError("Trajectoire inconnue")

    # True trajectory (deterministic: straight line at constant velocity)
    Xt = np.zeros((d, T))
    Xt[:, 0] = EtatVraiInitial
    for t in range(1, T):
        Xt[:, t] = F @ Xt[:, t - 1] + Commande[:, t]

    # Radar-altimeter measurements: h = z - h_terrain(x, y) + v
    R = SigmaBruit ** 2
    h_mes = Xt[2] - terr.hobs(Xt) + sqrt(R) * rng.standard_normal(T)

    # Filter-side measurement variance, inflated (tuned on the first scenario)
    R_f = (coefR ** 2) * R
    # MODIFICATION TP4 : la campagne peut fournir un jeu de données commun.
    # Xt sert à l'évaluation ; seuls h_mes et CentreNuageInitial corrigent le filtre.
    if Donnees is not None:
        Xt = np.asarray(Donnees["Xt"])
        h_mes = np.asarray(Donnees["h_mes"])
        Commande = np.asarray(Donnees["Commande"])
        if Xt.shape != (d,T) or h_mes.shape != (T,) or Commande.shape != (d,T):
            raise ValueError("Dimensions des données incompatibles")

    # Optimal bandwidth (Gaussian target), times the over-smoothing factor mu
    if ChoixNoyau == 1:      # Epanechnikov
        cd = pi ** (d / 2) / gamma(d / 2 + 1)
        KA = ((8 / cd) * (d + 4) * (2 * sqrt(pi)) ** d) ** (1 / (d + 4))
        h_RPF = mu * KA * Np ** (-1 / (d + 4))
    else:                    # Gaussian
        h_RPF = mu * (4 / (Np * (d + 2))) ** (1 / (d + 4))

    # MODIFICATION TP4 : les 2 noyaux d'origine utilisent un flux dédié.
    def kernel(n):
        if ChoixNoyau == 1:
            return gene_epanech(d, n, rng_reg)
        return rng_reg.standard_normal((d, n))

    # Initialisation : centre du nuage tiré uniformément dans un carré,
    # particules gaussiennes autour de ce centre, de covariance P0
    Xe = np.zeros((d, T))
    Xstd = np.zeros((d, T))                 # weighted particle std (posterior)
    CentreNuageInitial = EtatVraiInitial + RacineP0 @ (2 * rng.random(d) - 1)
    # MODIFICATION TP4 : centre incertain commun aux filtres de chaque cas.
    if Donnees is not None:
        CentreNuageInitial = np.asarray(Donnees["CentreNuageInitial"])
    # MODIFICATION TP4 : même covariance P0 pour les deux lois initiales.
    if LoiInitiale == "uniforme":
        Xp = CentreNuageInitial[:, None] + RacineP0 @ rng_init.uniform(-sqrt(3), sqrt(3), (d,Np))
    else:
        Xp = CentreNuageInitial[:, None] + RacineP0 @ rng_init.standard_normal((d,Np))
    p = np.full(Np, 1.0 / Np)
    Xe[:, 0] = Xp @ p
    Xstd[:, 0] = np.sqrt(((Xp - Xe[:, 0][:, None]) ** 2) @ p)

    Neff_hist = np.full(T, np.nan)
    Neff_hist[0] = Np
    resampled = np.zeros(T, dtype=bool)

    # MODIFICATION TP4 : sorties supplémentaires pour les comparaisons.
    Xmap = np.full((d,T), np.nan)
    DernierMAP = None
    PerteTotale = 0
    for t in range(1, T):
        # Prediction
        # MODIFICATION TP4 : commande connue et bruit ajustable pour la BCR.
        Xp = F @ Xp + Commande[:,t,None] + BruitDynamique * LQ @ rng_dyn.standard_normal((d, Np))
        # Correction (log-domain for numerical safety)
        inno = h_mes[t] - Xp[2] + terr.hobs(Xp)
        # MODIFICATION TP4 : invalider les cellules absentes/hors carte,
        # plutôt que prendre le relief du bord par le clipping de hobs.
        valides = (np.isfinite(inno) & (Xp[0]>=0) & (Xp[1]>=0)
                   & (Xp[0] < (terr.h_MNT.shape[1]-1)*terr.pasx)
                   & (Xp[1] < (terr.nrow-1)*terr.pasy))
        with np.errstate(divide="ignore"):
            logw = np.log(p) - inno ** 2 / (2 * R_f)
        logw[~valides] = -np.inf
        if not np.isfinite(logw.max()):
            # MODIFICATION TP4 : perte conservée dans la RMSE, sans recentrage.
            PerteTotale = t
            for j in range(t,T):
                Xe[:,j] = F @ Xe[:,j-1] + Commande[:,j]
            Xstd[:,t:] = np.nan
            Neff_hist[t:] = 0
            break
        logw -= logw.max()
        p = np.exp(logw)
        p /= p.sum()
        Xe[:, t] = Xp @ p
        Xstd[:, t] = np.sqrt(((Xp - Xe[:, t][:, None]) ** 2) @ p)

        Neff = 1.0 / np.sum(p ** 2)
        Neff_hist[t] = Neff
        msg = f"Obs. {t + 1:3d}: Neff = {Neff:.4f}"
        # MODIFICATION TP4 : covariance et MAP avant rééchantillonnage.
        if PasMAP and t % PasMAP == 0:
            Xc = Xp - Xe[:,t,None]
            At = (Xc*p) @ Xc.T
            DernierMAP = kde_map(Xp,p,Xe[:,t],At,DernierMAP)
            Xmap[:,t] = DernierMAP
        if Neff < Nth:
            Xc = Xp - Xe[:, t][:, None]
            At = (Xc * p) @ Xc.T
            # MODIFICATION TP4 : SIR = même sélection, sans noyau ajouté.
            # La Cholesky du professeur reste le calcul de référence du RPF.
            if TypeFiltre == "RPF":
                try:
                    LA = np.linalg.cholesky(At)
                except np.linalg.LinAlgError:
                    # MODIFICATION TP4 : secours pour une covariance singulière.
                    LA = psd_sqrt(At)
                Xp = Xp[:, select(p, rng_resamp)] + h_RPF * LA @ kernel(Np)
            else:
                Xp = Xp[:, select(p, rng_resamp)]
            p = np.full(Np, 1.0 / Np)
            resampled[t] = True
            msg += " resampling"
        if verbose:
            print(msg)

    # MODIFICATION TP4 : t=0 correspond à l'a priori, mesures à t=1,...,720.
    temps = np.arange(T) * dt
    err = Xt - Xe

    # MODIFICATION TP4 : retour détaillé pour lancer_tp.py, sans affichage.
    if RetourDetails:
        return dict(Xt=Xt, Xe=Xe, Xstd=Xstd, Xmap=Xmap, h_mes=h_mes,
                    CentreNuageInitial=CentreNuageInitial, Neff_hist=Neff_hist,
                    resampled=resampled, PerteTotale=PerteTotale)
    if not plot:
        return Xt, Xe

    # ================================================================ plots
    ext = terr.extent_km()

    # --- Fig. 1: map, full view + zoom on the trajectory -------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    for a in (a1, a2):
        im = a.imshow(terr.h_MNT, extent=ext, origin="upper", aspect="equal", cmap="terrain")
        a.plot(Xt[0] / 1000, Xt[1] / 1000, "-k", lw=1.5, label="vraie")
        a.plot(Xe[0] / 1000, Xe[1] / 1000, ".r", ms=3, label="estimée")
        a.plot(Xt[0, 0] / 1000, Xt[1, 0] / 1000, "ok", mfc="none", ms=8, label="départ")
        a.set_xlabel("km"); a.set_ylabel("km"); a.grid(True)
    a1.set_title("44° N, 3° E")
    pad = 5.0
    a2.set_xlim(Xt[0].min() / 1000 - pad, Xt[0].max() / 1000 + pad)
    a2.set_ylim(Xt[1].max() / 1000 + pad, Xt[1].min() / 1000 - pad)
    a2.set_title("zoom sur la trajectoire"); a2.legend(loc="lower left")
    fig.colorbar(im, ax=(a1, a2), label="altitude du relief (m)", shrink=0.8)

    # --- Fig. 2: errors with ±3σ particle envelopes ------------------------
    labels = ["x (m)", "y (m)", "z (m)", "vx (m/s)", "vy (m/s)", "vz (m/s)"]
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i, a in enumerate(axs.T.ravel()):
        a.plot(temps, err[i], "b", label="erreur réelle")
        a.plot(temps, 3 * Xstd[i], "r", lw=0.8, label="±3σ particules")
        a.plot(temps, -3 * Xstd[i], "r", lw=0.8)
        a.set_ylabel(labels[i]); a.grid(True)
        s = np.nanpercentile(np.abs(np.r_[err[i, 20:], 3 * Xstd[i, 20:]]), 99)
        a.set_ylim(-1.5 * s, 1.5 * s)      # zoom on the post-transient part
    for a in axs[-1]:
        a.set_xlabel("temps (s)")
    axs[0, 0].legend(loc="upper right")
    fig.suptitle("Erreur réelle (bleu) et écart-type des particules ±3σ (rouge)")
    fig.tight_layout()

    # ---------------------------------------------------------------- summary
    tail = slice(T // 2, T)
    print("\nRésumé")
    print(f"  h_RPF = {h_RPF:.4f}, rééchantillonnages : {resampled.sum()}/{T - 1}")
    names = ["x", "y", "z", "vx", "vy", "vz"]
    print("  erreur finale      :", ", ".join(f"{n}={e:+.2f}" for n, e in zip(names, err[:, -1])))
    print("  RMSE (2e moitié)   :", ", ".join(
        f"{n}={np.sqrt(np.mean(err[i, tail] ** 2)):.2f}" for i, n in enumerate(names)))
    return Xt, Xe


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1]:
        mat = sys.argv[1]
    else:
        # look for the map next to this script (Spyder: Run without arguments)
        here = os.path.dirname(os.path.abspath(__file__))
        mat = next((os.path.join(here, f) for f in ("carte.mat", "carte_2.mat")
                    if os.path.exists(os.path.join(here, f))), "carte.mat")
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(mat, sd)
    plt.show()
