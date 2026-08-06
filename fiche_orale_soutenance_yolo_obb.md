# Fiche orale — Soutenance YOLO26n-OBB

## Fil conducteur

Objets aériens petits et orientés → boîtes OBB → YOLO en une étape → IoU orientée → validation → comparaison avec Faster R-CNN.

## 0:00–1:00 — Pourquoi ce modèle ?

- DOTA contient 15 classes et beaucoup d'objets inclinés.
- Une HBB contient souvent trop d'arrière-plan ; une OBB suit l'orientation.
- YOLO détecte en une étape et reste rapide.
- La variante nano est réaliste sur une RTX 4060.
- 1024 pixels préserve mieux les petits détails que 640.

Phrase de sortie :

> Le choix est motivé par la géométrie de DOTA, puis confirmé par les résultats de validation.

## 1:00–4:00 — Les cinq étapes

1. Tuile d'entrée en 1024 × 1024.
2. Backbone CNN : extraction des caractéristiques.
3. Neck : fusion de plusieurs échelles.
4. Head OBB : classe, confiance et boîte orientée.
5. Seuil de confiance puis NMS orientée pour retirer les doublons.

Transition :

> Pour comparer les boîtes et mesurer leur superposition, on utilise l'IoU.

## 4:00–7:00 — IoU orientée

$$IoU(A,B)=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}$$

- Numérateur : aire commune.
- Dénominateur : aire totale couverte.
- 0 = aucune superposition ; 1 = superposition parfaite.
- Pour une OBB, la formule est la même, mais l'intersection est celle de polygones inclinés.

Exemple : deux aires de 100, intersection de 60 → union de 140 → IoU de 0,43.

## 7:00–8:00 — Résultats

| Modèle | F1 | mAP50-95 | Temps/image |
|---|---:|---:|---:|
| Faster R-CNN HBB | 0,213 | 0,060 | 17,1 ms |
| YOLO26n-OBB-1024 | 0,386 | 0,185 | 7,9 ms |

Interprétation : YOLO-OBB est meilleur et plus rapide dans cette expérience.

Limite à reconnaître : rappel = 0,291, donc beaucoup d'objets restent manqués.

## 8:00–9:15 — Démonstration

- Ouvrir `preparation_soutenance_yolo_obb.ipynb` à la section 10.
- Exécuter uniquement la cellule marquée `demo-soutenance`.
- Montrer le poids chargé, l'image de validation, les scores et les OBB.
- Si l'exécution échoue, descendre immédiatement à l'image de secours.

Phrase à dire :

> Je ne lance aucun entraînement. Je charge le meilleur poids déjà appris et j'effectue une inférence sur une tuile fixe de validation.

## 9:15–9:40 — Conclusion

> Les boîtes orientées sont cohérentes avec DOTA et YOLO26n-OBB-1024 dépasse ma baseline. Le pipeline fonctionne de bout en bout, mais le rappel montre qu'il reste une marge d'amélioration.

## Réponses flash

**Confiance ?** Croyance du modèle dans une prédiction.

**IoU ?** Superposition géométrique entre deux boîtes.

**NMS ?** Retire les prédictions redondantes en gardant les plus confiantes.

**mAP50 ?** Performance moyenne avec IoU minimale de 0,50.

**mAP50-95 ?** Moyenne sur plusieurs seuils plus exigeants.

**Pourquoi OBB ?** Meilleure représentation des objets inclinés.

**Pourquoi 1024 ?** Plus de détails sur les petits objets, au prix de plus de calcul.

**Pourquoi une baseline ?** Vérifier que le modèle principal apporte réellement un gain.

**YOLO toujours supérieur ?** Non, seulement dans cette expérience et avec ces réglages.

**Limite principale ?** Rappel faible et sous-ensemble expérimental.
