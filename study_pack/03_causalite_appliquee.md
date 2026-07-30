# Causalite appliquee aux erreurs du detecteur

## 1. Prediction, association et causalite

### Prediction

Question :

> Peut-on predire si cet objet sera correctement detecte ?

Une forte performance predictive signifie que les variables disponibles
contiennent de l'information sur l'outcome. Elle ne dit pas ce qui arriverait
si on changeait une variable.

### Association

Question :

> Le taux de detection est-il different entre petits et grands objets ?

Une difference observee peut venir de la taille, mais aussi des classes, de la
resolution, de la densite, de la source ou d'autres facteurs.

### Causalite

Question :

> Pour des objets comparables, que deviendrait la probabilite de detection si
> l'objet etait dans la condition "tres petit" plutot que "non tres petit" ?

Cette question compare deux mondes potentiels. Un seul est observe pour chaque
objet. L'autre est contrefactuel.

## 2. Cadre des outcomes potentiels

Pour l'objet `i` :

```text
Y_i(1) = outcome si l'objet est tres petit
Y_i(0) = outcome si l'objet n'est pas tres petit
```

Effet individuel :

```text
tau_i = Y_i(1) - Y_i(0)
```

Effet moyen :

```text
ATE = E[Y(1) - Y(0)]
```

On n'observe jamais les deux outcomes pour le meme objet. C'est le probleme
fondamental de l'inference causale. Les methodes utilisent d'autres objets
comparables et des hypotheses d'identification.

## 3. Question retenue dans DOTA

Traitement `D` :

- `D = 1` si l'aire relative de l'objet dans sa tuile est inferieure ou egale
  au premier quartile calcule sur les objets train uniques ;
- `D = 0` sinon.

Outcome `Y` :

- `Y = 1` si le modele OBB associe une prediction de meme classe, de confiance
  au moins 0.25 et d'IoU orientee au moins 0.50 ;
- `Y = 0` sinon.

Estimand :

- effet moyen du traitement dans la population des objets uniques du
  sous-ensemble de validation.

Question :

> Quel est l'effet d'etre un objet tres petit sur la probabilite d'une
> detection correcte, apres ajustement sur la classe, l'orientation, la densite,
> la source, le GSD, le ratio de forme et les conditions de tuilage ?

## 4. Prudence sur le mot traitement

"Rendre un objet petit" n'est pas une intervention physique unique :

- eloigner le capteur ;
- diminuer la resolution ;
- changer la taille reelle de l'objet ;
- redimensionner l'image.

Ces interventions pourraient avoir des effets differents. Le traitement est
donc une condition observationnelle operationnelle. L'interpretation causale
exige de supposer qu'une intervention pertinente sur la taille apparente est
bien definie.

Une formulation prudente :

> Sous les hypotheses d'ajustement et pour cette definition operationnelle, on
> estime le contraste associe au passage dans la categorie tres petite.

## 5. DAG

Version simplifiee :

```text
Classe ----------> Taille tres petite ----------> Detection correcte
   |                       |                              ^
   |                       v                              |
   +-----------------> Difficulte -----------------------+

GSD -------------> Taille tres petite
 |
 +--------------------------------------------------> Detection correcte

Source ----------> GSD
  |                                                  ^
  +--------------------------------------------------+

Densite -------------------------------------------> Detection correcte
Orientation --------------------------------------> Detection correcte
Ratio de forme -----------------------------------> Detection correcte
Bord de tuile ------------------------------------> Detection correcte
```

Le DAG n'est pas une decoration. Il encode les chemins supposes et guide les
variables a ajuster.

## 6. Confounder

Un confounder est une cause commune du traitement et de l'outcome.

Exemple :

- certaines classes sont typiquement petites ;
- la classe influence aussi la facilite de detection.

Chemin non causal :

```text
Taille tres petite <- Classe -> Detection correcte
```

Sans ajuster sur la classe, une mauvaise performance des petits objets pourrait
etre en partie une difference entre classes.

Reponse orale :

> Un confounder cree un chemin arriere entre traitement et outcome. J'ajuste
> dessus pour comparer des objets plus semblables et bloquer cette association
> non causale.

## 7. Mediator

Un mediator est cause par le traitement et transmet une partie de son effet.

Exemple theorique :

```text
Taille tres petite -> Qualite des features -> Detection correcte
```

Si l'objectif est l'effet total de la petite taille, ajuster sur la qualite des
features bloquerait une partie de l'effet que l'on veut mesurer.

Le champ `difficult` peut en partie resumer les consequences de la taille et de
la visibilite. Il n'est donc pas inclus automatiquement dans l'ajustement
principal.

## 8. Collider

Un collider est une consequence commune de deux variables :

```text
Taille tres petite -> Selection <- Scene complexe
```

Si on conditionne sur `Selection`, on peut creer artificiellement une
association entre taille et complexite, meme si elles etaient independantes.

Exemple dans le projet :

- une tuile peut etre rejetee parce qu'un objet est coupe de facon ambigue ;
- la coupure depend de la position/taille de l'objet et de la grille ;
- analyser seulement des tuiles acceptees peut introduire une selection.

On ne peut pas toujours eliminer ce biais, mais il faut le declarer.

## 9. Backdoor path

Un backdoor path est un chemin entre `D` et `Y` qui commence par une fleche
entrant dans `D`.

Exemple :

```text
D <- Classe -> Y
```

L'ajustement cherche un ensemble de variables qui bloque les backdoors sans
bloquer les mediateurs ni ouvrir les colliders.

## 10. Exemple smoking, cancer et gene

DAG plausible :

```text
Gene -> Smoking -> Cancer
  \----------------> Cancer
```

Le gene est un confounder s'il influence a la fois smoking et cancer.

Pour estimer l'effet total de smoking sur cancer, on ajuste sur le gene. On
n'ajuste pas sur un mediator cause par smoking si on veut l'effet total.

Si on avait :

```text
Smoking -> Hospitalisation <- Cancer
```

Hospitalisation serait un collider. Conditionner dessus pourrait creer une
association trompeuse.

## 11. Variables d'ajustement du projet

### Incluses

- classe ;
- orientation absolue ;
- log du ratio de forme ;
- densite d'objets dans la tuile ;
- source d'image ;
- GSD ;
- marge au bord de tuile ;
- fraction de l'objet conservee.

### Non incluse dans l'ajustement principal

- aire exacte, parce qu'elle definit directement le traitement ;
- `difficult`, car elle peut resumer une consequence de la petite taille ;
- score de confiance, car il est produit par le detecteur apres le traitement ;
- IoU, car elle definit l'outcome ;
- predictions intermediaires du reseau, qui sont post-traitement.

## 12. Hypothese d'echangeabilite conditionnelle

Forme :

```text
(Y(0), Y(1)) independants de D conditionnellement a X
```

Intuition :

> Apres ajustement sur X, le groupe tres petit et le groupe controle sont
> comparables quant a leurs outcomes potentiels.

Cette hypothese ne peut pas etre prouvee avec les donnees. Elle est fragile ici
car des variables peuvent manquer :

- contraste local ;
- flou ;
- occultation ;
- conditions de capture ;
- qualite d'annotation ;
- texture du fond.

## 13. Positivite

Pour chaque combinaison de covariables pertinente :

```text
0 < P(D = 1 | X) < 1
```

Il faut des objets petits et non petits comparables. Si tous les helicopters
d'un certain type sont minuscules, leur contrefactuel non petit n'est pas
soutenu par les donnees.

Diagnostic :

- distributions des scores de propension ;
- fraction des scores entre 0.1 et 0.9 ;
- poids extremes ;
- effectifs par sous-groupe.

Clipper les scores limite la variance, mais ne cree pas de donnees manquantes.

## 14. Coherence et consistance

Si l'objet recoit effectivement `D=d`, son outcome observe doit correspondre a
`Y(d)`.

Cette hypothese suppose une version assez claire du traitement. Elle est
fragile lorsque plusieurs mecanismes differents produisent la meme categorie
"tres petit".

## 15. SUTVA et interference

SUTVA suppose notamment que l'outcome d'un objet ne depend pas du traitement
d'un autre objet.

Dans la detection, cette independance peut echouer :

- NMS entre objets voisins ;
- occlusion ;
- limite `max_det` dans une tuile dense ;
- contexte partage.

Le bootstrap par image reconnait la correlation statistique entre objets de la
meme image, mais il ne supprime pas une interference causale.

## 16. Absence de fuite

Les predictions causales doivent provenir d'un modele evalue hors de ses images
train. Une prediction sur un objet vu pendant l'entrainement aurait une erreur
trop optimiste.

Ici :

- le detecteur est ajuste sur les tuiles train ;
- l'outcome est construit sur les tuiles validation ;
- aucune image source n'est partagee.

## 17. Difference brute

```text
E[Y | D=1] - E[Y | D=0]
```

Avantage :

- simple et descriptif.

Limite :

- melange effet du traitement et differences de composition.

Elle reste utile comme point de comparaison. Si l'effet ajuste change beaucoup,
les covariables jouaient un role important.

## 18. Regression ajustee et g-computation

On modelise :

```text
mu_d(X) = E[Y | D=d, X]
```

Puis on predit pour chaque objet :

```text
mu_1(X_i) - mu_0(X_i)
```

et on moyenne.

Avantage :

- produit directement deux outcomes attendus.

Limite :

- biais si le modele d'outcome est mal specifie.

Dans le pipeline, deux forets de classification servent de T-learner : une pour
les traites, une pour les controles.

## 19. Score de propension

```text
e(X) = P(D=1 | X)
```

Il resume la probabilite d'etre traite compte tenu des covariables.

Le score ne prouve pas que les groupes sont comparables. Il aide a ajuster les
variables observees et a diagnostiquer la positivite.

## 20. IPW

Idee :

- un traite rare dans son profil recoit un poids proche de `1/e(X)` ;
- un controle rare recoit un poids proche de `1/(1-e(X))`.

L'echantillon pondere vise une pseudo-population ou le traitement est moins
associe aux covariables.

Limite :

- score proche de 0 ou 1 -> poids enorme -> variance et instabilite.

## 21. AIPW doublement robuste

Le score AIPW combine modele d'outcome et score de propension :

```text
psi_i =
mu_1(X_i) - mu_0(X_i)
+ D_i * (Y_i - mu_1(X_i)) / e(X_i)
- (1-D_i) * (Y_i - mu_0(X_i)) / (1-e(X_i))
```

Intuition :

1. g-computation fournit une prediction de base ;
2. les residus observes corrigent cette prediction ;
3. la correction utilise la probabilite de traitement.

"Doublement robuste" signifie que l'estimateur peut rester coherent si l'un des
deux modeles nuisances est correctement specifie, sous les autres hypotheses.
Cela ne protege pas contre :

- confounder non observe ;
- mauvaise definition du traitement ;
- absence de positivite ;
- fuite de donnees ;
- selection.

## 22. Cross-fitting

Si le meme objet sert a entrainer les modeles nuisances et a calculer son
pseudo-outcome, un modele flexible peut overfitter.

Le cross-fitting :

1. separe les images en folds ;
2. ajuste les modeles sur les autres folds ;
3. predit sur le fold tenu a l'ecart ;
4. rassemble les predictions hors fold.

Le split est groupe par image source pour ne pas separer des objets fortement
lies d'une meme image.

## 23. Intervalles de confiance

Les objets d'une meme image ne sont pas independants. Un bootstrap objet par
objet donnerait trop d'unites independantes.

Le bootstrap groupe :

1. reechantillonne les images avec remise ;
2. prend tous leurs objets ;
3. recalcule la moyenne des scores ;
4. utilise les quantiles 2.5 % et 97.5 %.

Limite du pipeline :

- les modeles nuisances ne sont pas re-ajustes dans chaque repetition, donc
  l'incertitude totale peut etre sous-estimee.

## 24. Effet heterogene

L'ATE peut masquer des differences :

```text
CATE(x) = E[Y(1) - Y(0) | X=x]
```

Exemples :

- effet plus negatif dans les scenes denses ;
- effet different selon la classe ;
- effet plus fort pour certaines orientations ;
- effet incertain pour classes rares.

Une heterogeneite estimee n'est pas automatiquement reelle. Chercher beaucoup
de sous-groupes augmente le risque de trouver du bruit.

## 25. Arbre causal honnete

Le pipeline utilise un arbre sur les pseudo-outcomes AIPW :

- moitie des images pour choisir les divisions ;
- autre moitie pour estimer les effets dans les feuilles ;
- profondeur limitee ;
- effectif minimal par feuille.

"Honnete" signifie que les memes observations ne choisissent pas les coupures
et n'estiment pas ensuite leurs effets. Cela reduit le biais de selection.

Une feuille de petite taille ou dominee par une seule classe doit etre
interpretee prudemment.

## 26. Foret sur pseudo-outcomes

Plusieurs arbres de regression apprennent `psi_i` a partir de `X`. Le pipeline
utilise un split croise par image afin que la CATE d'un objet provienne d'une
foret qui n'a pas utilise son image.

Cette methode est une approximation pedagogique d'une foret causale :

- elle utilise un score orthogonal AIPW ;
- elle permet d'explorer l'heterogeneite ;
- elle n'implemente pas toutes les garanties d'un package specialise de
  generalized random forest.

Il faut nommer cette limite, pas presenter l'outil comme magique.

## 27. Analyse de sensibilite

Le resultat principal depend de choix :

- IoU 0.50 pour definir une detection correcte ;
- seuil du premier quartile pour "tres petit" ;
- clipping du score de propension ;
- modele nuisances ;
- sous-ensemble.

Le pipeline varie au moins :

- IoU 0.40, 0.50 et 0.60 ;
- clipping 0.02, 0.05 et 0.10.

Un signe stable renforce la conclusion descriptive sous les hypotheses. Des
variations fortes signalent une conclusion fragile.

## 28. Comment interpreter un effet

Exemple fictif :

```text
ATE AIPW = -0.18
IC 95 % = [-0.25, -0.10]
```

Interpretation prudente :

> Sous les hypotheses d'identification, la condition tres petite est associee a
> une diminution moyenne estimee de 18 points de pourcentage de la probabilite
> de detection correcte. L'intervalle bootstrap exclut zero dans ce protocole.

Ne pas dire :

> Rendre n'importe quel objet petit fera toujours baisser la detection de 18 %.

L'effet est une difference de probabilite en points de pourcentage, pas
necessairement une baisse relative de 18 %.

## 29. Liste de limites a savoir expliquer

1. Traitement non randomise.
2. Variables non observees.
3. Intervention "taille" imparfaitement definie.
4. Sous-ensemble selectionne.
5. Classes rares et manque de positivite local.
6. Dependances entre objets d'une image.
7. NMS et interference possible.
8. Outcome produit par un seul detecteur.
9. Erreurs d'annotation.
10. Selection due au tuilage et aux fragments.
11. Seuils de confiance et IoU arbitraires mais analyses.
12. Foret causale approximative.
13. Bootstrap ne re-ajustant pas tous les modeles.
14. Generalisation limitee a la population analysee.

## 30. Questions de rappel

1. Pourquoi l'outcome potentiel non observe pose-t-il un probleme ?
2. Quelle est la population de l'ATE ici ?
3. Pourquoi la classe peut-elle etre un confounder ?
4. Pourquoi ne pas ajuster sur le score de confiance ?
5. Comment ajuster sur un mediator modifie-t-il l'estimand ?
6. Comment conditionner sur un collider cree-t-il un biais ?
7. Quelle difference entre exchangeabilite et positivite ?
8. Pourquoi le score de propension ne resout-il pas les confounders non observes ?
9. Pourquoi IPW devient-il instable ?
10. Que corrige la partie residuelle de l'AIPW ?
11. Pourquoi le cross-fitting est-il groupe par image ?
12. Pourquoi le bootstrap est-il groupe par image ?
13. Que signifie "honnete" pour l'arbre ?
14. Pourquoi une CATE extreme peut-elle etre du bruit ?
15. Quelle phrase permet d'interpreter un effet sans surpromettre ?
