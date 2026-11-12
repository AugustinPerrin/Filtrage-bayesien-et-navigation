# TP4 

## Lancer le fichier 

Placer `carte.mat` et `outils_tp.py` à côté, puis dans Spyder ouvrir `rpf_altimetrie.py` et lancer F5. Dans un terminal :

```bash
python rpf_altimetrie.py carte.mat 0
```

Les valeurs par défaut sont `mu=0.3`, `coefR=1.5` et `CoefResampling=0.5`. Pour reproduire les paramètres du rapport sur un seul essai :

```python
from rpf_altimetrie import main
Xt, Xe = main('carte.mat', seed=0, NbPart=5000, mu=0.4,
              CoefResampling=0.4, coefR=1.0,
              LoiInitiale='gauss', TypeFiltre='RPF',
              PositionInitiale=(150000., 150000.), Trajectoire='droite')
```

`LoiInitiale='uniforme'` change la loi du nuage initial. `ChoixNoyau=1` utilise bien le noyau d'Epanechnikov et appelle `gene_epanech` ; `ChoixNoyau=2` utilise le noyau gaussien. Ces 2 choix sont distincts. `TypeFiltre='SIR'` supprime seulement l'ajout du noyau lors du rééchantillonnage. `Trajectoire='boucle'` active la commande de virage connue.


Le script `lancer_tp.py` appelle directement `main` dans `rpf_altimetrie.py`** pour chaque essai. Il fournit les mêmes mesures et le même centre initial aux filtres comparés. Les 100 cas ont des bruits et erreurs initiales différents. 

## Fichiers

- `rpf_altimetrie.py` : fichier du professeur adapté ; filtre, rééchantillonnage, 2 noyaux et affichage d'origine.
- `lancer_tp.py` : organisation des 100 cas, RMSE et figures comparatives.
- `outils_tp.py` : génération des données communes, boucle, gradient du terrain pour la BCR et MAP approché. **Pas de filtre dans ce fichier.**
- `resultats/` et `figures/` : données, tableaux et figures nécessaires au rapport.


