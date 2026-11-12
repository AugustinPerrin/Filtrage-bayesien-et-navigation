"""python lancer_tp.py --calcul --figures
24 configurations, 100 cas par configuration. Sorties existantes réutilisées.
Pour tout recalculer : --force ; pour limiter le parallélisme : --workers 2.
"""

import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
from pathlib import Path
import argparse,json,csv
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from outils_tp import *
from rpf_altimetrie import main
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'resultats';FIG=ROOT/'figures'
ZONES={'vallonne':(150000.,150000.),'plat':(70000.,290000.)}

def configurations():
    # Tuple : zone, loi initiale, filtre, mu, N, trajectoire, experience BCR.
    cfg=[]
    for zone in ZONES:
        for method,mu in [('SIR',0.),('RPF',.4)]:cfg.append((zone,'gauss',method,mu,5000,'droite',False))
    for method,mu in [('SIR',0.),('RPF',.4)]:cfg.append(('vallonne','uniforme',method,mu,5000,'droite',False))
    for zone in ZONES:
        for mu in (.3,.5,.7):cfg.append((zone,'gauss','RPF',mu,5000,'droite',False))
    for n in (500,1000,2000):
        for method,mu in [('SIR',0.),('RPF',.4)]:cfg.append(('vallonne','gauss',method,mu,n,'droite',False))
    for zone in ZONES:
        for method,mu in [('SIR',0.),('RPF',.4)]:cfg.append((zone,'gauss',method,mu,5000,'boucle',False))
    for method,mu in [('SIR',0.),('RPF',.4)]:cfg.append(('vallonne','gauss',method,mu,5000,'droite',True))
    return cfg

def key(c):
    z,l,f,mu,n,tr,b=c;return f'{z}_{l}_{f}_mu{mu:g}_N{n}_{tr}_B{int(b)}'

def job(c,m):
    z,law,method,mu,n,tr,bayes=c
    terr=Terrain();initial=nominal(*ZONES[z]);rng=np.random.default_rng(12000+m)
    if bayes:center=initial.copy();initial=prior_sample(center,1,'gauss',rng)[:,0]
    else:center=initial+SIGMA0*rng.uniform(-1,1,6)
    u=loop_command() if tr=='boucle' else np.zeros((6,T+1))
    truth,y=simulate(terr,initial,rng,command=u)
    donnees=dict(Xt=truth,h_mes=y,CentreNuageInitial=center,Commande=u)
    out=main(ROOT/'carte.mat',seed=23000+m,NbPart=n,mu=mu,
             LoiInitiale=law,TypeFiltre=method,CoefResampling=.4,coefR=1.,
             BruitDynamique=0. if bayes else 1.,Donnees=donnees,
             plot=False,verbose=False,RetourDetails=True)
    return m,(out['Xe']-truth).astype('float32'),out['resampled'].sum(),out['PerteTotale']

def compute(workers=6,force=False):
    OUT.mkdir(exist_ok=True);cfg=configurations();manifest=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,c in enumerate(cfg):
            dest=OUT/(key(c)+'.npz');manifest.append(dict(key=key(c),configuration=c,M=100))
            if dest.exists() and not force:continue
            err=np.empty((100,6,T+1),np.float32);nr=np.empty(100);lost=np.empty(100)
            for f in as_completed([pool.submit(job,c,m) for m in range(100)]):
                m,e,n,l=f.result();err[m]=e;nr[m]=n;lost[m]=l
            e2=err.astype(float)**2
            rmse=np.sqrt(e2.mean(axis=0));horizontal=np.sqrt(e2[:,:2].sum(axis=1).mean(axis=0))
            trial=np.sqrt(e2[:,:2,360:].sum(axis=1).mean(axis=1))
            np.savez_compressed(dest,rmse=rmse,horizontal=horizontal,trial=trial,
                                final=np.linalg.norm(err[:,:2,-1],axis=1),resampled=nr,lost=lost)
            print(f'{i+1}/24 {key(c)} E_H={np.sqrt(np.mean(trial**2)):.1f} m',flush=True)
    (OUT/'protocole.json').write_text(json.dumps({'M':100,'configurations':manifest,
        'graine_donnees':'12000+m','graine_filtre':'23000+m','alpha':.4,'sigma_mesure':30,
        'numpy':np.__version__},indent=2))

def read(c):return np.load(OUT/(key(c)+'.npz'))

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'axes.grid':True,'grid.alpha':.2,'savefig.dpi':170})
    colors={'SIR':'#b6523a','RPF':'#166c92'};t=np.arange(T+1)
    def save(fig,name):
        fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(FIG/(name+'.'+ext),bbox_inches='tight')
        plt.close(fig)
    rows=[]
    for c in configurations():
        a=read(c)
        rows.append(dict(zone=c[0],loi=c[1],filtre=c[2],mu=c[3],N=c[4],trajectoire=c[5],BCR=int(c[6]),
                         EH=float(np.sqrt(np.mean(a['trial']**2))),final=float(a['horizontal'][-1]),
                         mediane=float(np.median(a['final'])),echecs=int(np.sum(a['final']>1000)),
                         pertes=int(np.sum(a['lost']>0)),redistributions=float(np.mean(a['resampled']))))
    with open(OUT/'synthese.csv','w') as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    # 4 trajets avec exactement les mêmes durée, départs et vitesse nominale.
    terr=Terrain();fig,axs=plt.subplots(1,3,figsize=(11,3.6));terrain_stats=[]
    im=axs[0].imshow(terr.z,origin='lower',cmap='terrain',extent=[0,(terr.nx-1)*terr.dx/1000,0,(terr.ny-1)*terr.dy/1000])
    for iz,(zone,(x,y)) in enumerate(ZONES.items()):
        for tr in ('droite','boucle'):
            u=loop_command() if tr=='boucle' else None
            X,_=simulate(terr,nominal(x,y),np.random.default_rng(0),command=u)
            h,hx,hy=terr.evaluate(X[0],X[1],True)
            terrain_stats.append(dict(zone=zone,trajectoire=tr,ecart_type=float(np.std(h)),pente_rms=float(np.sqrt(np.mean(hx*hx+hy*hy)))))
            axs[0].plot(X[0]/1000,X[1]/1000,label=zone+' / '+tr)
            axs[iz+1].plot(t,h,label=tr)
        axs[iz+1].set(title=zone,xlabel='Temps (s)',ylabel='Altitude du terrain (m)');axs[iz+1].legend()
    axs[0].set(xlabel='x (km)',ylabel='y (km)',title='MNT et trajectoires');axs[0].legend(fontsize=6)
    save(fig,'zones');(OUT/'terrain.json').write_text(json.dumps(terrain_stats,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,zone in zip(axs,ZONES):
        for meth,mu in [('SIR',0),('RPF',.4)]:
            a=read((zone,'gauss',meth,mu,5000,'droite',False))
            ax.semilogy(t,a['horizontal'],label=meth,color=colors[meth])
        ax.set(title=zone,xlabel='Temps (s)',ylabel='RMSE horizontale (m)');ax.legend()
    save(fig,'comparaison')
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,zone in zip(axs,ZONES):
        for mu in (.3,.4,.5,.7):
            a=read((zone,'gauss','RPF',mu,5000,'droite',False));ax.semilogy(t,a['horizontal'],label=f'μ={mu:g}')
        ax.set(title=zone,xlabel='Temps (s)',ylabel='RMSE horizontale (m)');ax.legend(fontsize=8)
    save(fig,'mu')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for ax,method,mu in zip(axs,['SIR','RPF'],[0,.4]):
        for n in (500,1000,2000,5000):
            a=read(('vallonne','gauss',method,mu,n,'droite',False));ax.semilogy(t,a['horizontal'],label=f'N={n}')
        ax.set(title=method,xlabel='Temps (s)',ylabel='RMSE horizontale (m)');ax.legend(fontsize=8)
    save(fig,'particules')
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,zone in zip(axs,ZONES):
        for method,mu in [('SIR',0),('RPF',.4)]:
            for tr,ls in [('droite','--'),('boucle','-')]:
                a=read((zone,'gauss',method,mu,5000,tr,False))
                ax.semilogy(t,a['horizontal'],ls,color=colors[method],label=method+' / '+tr)
        ax.set(title=zone,xlabel='Temps (s)',ylabel='RMSE horizontale (m)');ax.legend(fontsize=8)
    save(fig,'boucles')
    bp=OUT/'bcr.npz'
    if not bp.exists():
        B,_=bayesian_bound(terr,nominal());np.savez_compressed(bp,bound=B)
    B=np.load(bp)['bound'];BH=np.sqrt(B[:,0,0]+B[:,1,1]);ratio={}
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for method,mu in [('SIR',0),('RPF',.4)]:
        a=read(('vallonne','gauss',method,mu,5000,'droite',True))
        axs[0].semilogy(t,a['horizontal'],color=colors[method],label=method)
        rho=a['horizontal']/BH;axs[1].semilogy(t,rho,color=colors[method],label=method)
        # 1er instant >=30s où le ratio reste <=2 jusqu'à la fin.
        good=np.maximum.accumulate(rho[::-1])[::-1]<=2
        ks=np.flatnonzero(good & (t>=30))
        ratio[method]={'temps_ratio2':int(ks[0]) if len(ks) else None,
            'ratio_aux_temps':{str(k):float(rho[k]) for k in [30,60,180,360,720]}}
    axs[0].semilogy(t,BH,'k--',label='Racine BCR horizontale');axs[0].set(xlabel='Temps (s)',ylabel='Erreur horizontale (m)');axs[0].legend(fontsize=8)
    axs[1].axhline(2,c='black',ls='--',label='Rapport 2');axs[1].set(xlabel='Temps (s)',ylabel='RMSE / racine BCR');axs[1].legend(fontsize=8)
    save(fig,'borne');(OUT/'rapprochement_bcr.json').write_text(json.dumps(ratio,indent=2))
    # Même réalisation qu'au rapport initial, sans accès du filtre à la vérité.
    rng=np.random.default_rng(12000);initial=nominal();center=initial+SIGMA0*rng.uniform(-1,1,6)
    X,y=simulate(terr,initial,rng)
    o=main(ROOT/'carte.mat',seed=23000,mu=.4,CoefResampling=.4,coefR=1.,
           Donnees=dict(Xt=X,h_mes=y,CentreNuageInitial=center,Commande=np.zeros_like(X)),
           RetourDetails=True,PasMAP=5,plot=False,verbose=False)
    a=dict(mean=o['Xe'],map=o['Xmap'])
    metrics={}
    kk=np.arange(5,T+1,5)
    for est in ('mean','map'):
        ee=a[est][:2,kk]-X[:2,kk]
        metrics[est]=float(np.sqrt(np.mean(np.sum(np.diff(ee,axis=1)**2,axis=0))))
    (OUT/'estimateurs.json').write_text(json.dumps(metrics,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(10,3.5));k=np.arange(5,T+1,5)
    for i,ax in enumerate(axs):
        for est,label in [('mean','MMSE'),('map','MAP approché')]:ax.plot(k,a[est][i,k]-X[i,k],label=label)
        ax.set(xlabel='Temps (s)',ylabel='Erreur '+['x','y'][i]+' (m)');ax.legend(fontsize=8)
    save(fig,'estimateurs')

    def table(name,selected,fmt):
        chosen=[r for r in rows if selected(r)]
        if name=='tabparticules':chosen.sort(key=lambda r:(r['N'],r['filtre']))
        if name=='tabmu':chosen.sort(key=lambda r:(r['zone'],r['mu']))
        lines=[fmt(r)+r' \\' for r in chosen]
        (OUT/(name+'.tex')).write_text('\\def\\'+name+'{\n'+'\n'.join(lines)+'\n}\n')
    table('tablois',lambda r:r['zone']=='vallonne' and r['N']==5000 and r['trajectoire']=='droite' and not r['BCR'] and r['mu'] in (0,.4),
          lambda r:f"{r['loi']} & {r['filtre']} & {r['EH']:.1f} & {r['echecs']}")
    table('tabmu',lambda r:r['loi']=='gauss' and r['N']==5000 and r['trajectoire']=='droite' and not r['BCR'] and r['filtre']=='RPF',
          lambda r:f"{r['zone']} & {r['mu']:.1f} & {r['EH']:.1f} & {r['echecs']}")
    table('tabparticules',lambda r:r['zone']=='vallonne' and r['loi']=='gauss' and r['trajectoire']=='droite' and not r['BCR'] and r['mu'] in (0,.4),
          lambda r:f"{r['N']} & {r['filtre']} & {r['EH']:.1f} & {r['echecs']}")
    table('tabboucles',lambda r:r['loi']=='gauss' and r['N']==5000 and r['mu'] in (0,.4) and not r['BCR'],
          lambda r:f"{r['zone']} & {r['trajectoire']} & {r['filtre']} & {r['EH']:.1f} & {r['echecs']}")
    print('Figures et synthèse générées.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--calcul',action='store_true');ap.add_argument('--figures',action='store_true');ap.add_argument('--force',action='store_true');ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args()
    if not a.calcul and not a.figures:a.calcul=a.figures=True
    if a.calcul:compute(a.workers,a.force)
    if a.figures:figures()
