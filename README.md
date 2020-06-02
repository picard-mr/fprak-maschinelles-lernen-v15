# F-Praktikumsversuch 15 - Maschinelles Lernen in der wissenschaftlichen Bildanalyse
In diesem Versuch sollen Grundprinzipien der Bildsegmentierung von Mikroskopbildern mithilfe von *Deep Learning* vermittelt werden. Dazu werden Intensitäten-basierte Verfahren mit den Verfahren auf Grundlage von *Neuronalen Netzwerken* verglichen. *Neuronale Netzwerke* bilden eine Klasse von Lern-Algorithmen welche auch unter schlechten Signal-zu-Rausch-Verhältnissen gute Segmentierungen poduzieren können [[1](https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-019-2880-8)]. Die Segmentierung von Mikroskopbildern ist eine Grundvoraussetzung um physikalische Phänomene innerhalb von bakteriellen Kolonien quantitativ zu beschreiben. Die Kenntnisse über verschiedene Segmentierungsverfahren gehören zum Grundlagen der Bildanalyse und sind unverzichtbar in der Auswertung von Mikroskopieexperimenten. Für diesen Versuch werden werden Programmierkenntnisse in Python benötigt.

Neben theoretischen Kenntnissen sollen auch die praktischen Fertigkeiten bei der Erstellung von wissenschaftlicher Software geschult werden. Dazu werden einige kleinere Werkzeuge/ Methoden vorgestellt, die das Programmieren erleichtern bzw. wiederkehrende Aufgaben automatisieren können und sich in der Praxis bewährt haben.

## Aufbau und weitere Informationen
### Begrifflichkeiten
Für einen reibunglosen Ablauf werden Kenntnisse über folgende Begrifflichkeiten vorausgesetzt:
- *U-Net* als Beispiel eines *Convolutional Neuronal Network* / *Auto-Encoder*
- *Semantical* vs. *Instance Segmentation*
- Unterscheidung von *Unsupervised* und *Supervised Learning*
- *Loss Function*
- Repräsentationen von Formen / Objekten in Computern

### Paper
Dieser F-Praktikumsversuch nimmt Bezug auf verschiedene wissenschaftliche Paper:
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Olaf Ronneberger, Philipp Fischer, Thomas Brox, 2015
- Die zwei [stardist Paper](https://github.com/mpicbg-csbd/stardist) und ihre Implementierung auf Github.
- [Noise2Void - Learning Denoising from Single Noisy Images](https://arxiv.org/abs/1811.10980), Alexander Krull, Tim-Oliver Buchholz, Florian Jug, 2019 und Implementierung auf [Github](https://github.com/juglab/n2v)

### Frameworks
Im Versuch wird das Deep Learning Framework [Keras](https://keras.io/) mit Backend [tensorflow](https://www.tensorflow.org/) verwendet. Fortgeschrittene Studierende, die sich mit anderen Frameworks (bspw. [PyTorch](https://pytorch.org/)) auskennen, können den Versuch gerne auch mit anderen Frameworks durchführen.

### Ablauf
Die hier verfügbaren Notebooks sollen bearbeitet werden. Dafür können die Studierenden einen PC ihrer Wahl verwenden. Allerdings werden GPUs für viele Berechnungen benötigt. Daher haben Studierende die Möglichkeit die Student-Workstation [Picard](https://picard.physik.uni-marburg.de) zu verwenden. Mit [diesem Link](https://picard.physik.uni-marburg.de/jupyterhub/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fpicard-mr%2Ffprak-maschinelles-lernen-v15&urlpath=tree%2Ffprak-maschinelles-lernen-v15%2F) wird dieses Git Repository in das Home-Verzeichnis auf Picard kopiert und kann dann bearbeitet werden. Studierende, die dieses Repository ausführlicher nutzen wollen und sich bereits mit Git besser auskennen, können dieses Repository auch forken oder per SSH / HTTPS klonen.

Die Notebooks sind nummeriert und sollen in dieser Reihenfolge auch abgearbeitet werden:
1. **Die Daten:** Macht euch mit den Daten vertraut. Implementierung eines Viewers. Vorbereiten der Daten.
5. **Otsu-Thresholding / Kantenfinden:** Ausprobieren klassischer Filter-Methoden.
10. **Entrauschen:** Noise2Void Training.
20. **Instance Segmentation:** Stardist Training.
30. **Implementierung U-Net / ResNet:** Eigene Implementierung zur Semantic Segmentation. Anschließendes Training.
40. **Validierung:** Die trainierten Modelle sollen validiert werden.

Dieser Versuch erfordert nicht, dass die Studierenden körperlich anwesend sind. Erfahrungsgemäß lassen sich Probleme und Fragen jedoch einfacher persönlich besprechen. Die Studierenden werden aufgefordert einen eigenen Laptop mit zum Versuch zu bringen. In der eigenen Umgebung ist man schließlich am effektivsten. Falls dies nicht möglich ist, werden PCs für den Versuch gestellt.

### Aufbau des Versuchsprotokolls
Generell sollten alle ausgeführten Aufgaben protokolliert werden und die verschiedenen Methoden mit einander verglichen werden. Auf einen Grundlagenteil, der die verwendeten Methoden erläutert, wird Wert gelegt.

### Vereinfachungen
Während in unserer Arbeitsgruppe auch 3-dimensionale Mikroskop-Bilder analysiert werden, werden in diesem Versuch nur 2-dimensionale Mikroskop-Bilder analysiert.

## Weitere Tools
### Jupyter
Fast der gesamte Versuch wird in einem Jupyter Notebook durchgeführt. Das [Project Juypter](https://jupyter.org/) ist sehr beeindruckend und kann das Entwickeln und Teilen von Code, sowie das Programmieren Lernen sehr erleichtern.
### Git
Die Versionskontroll-Software [Git](https://git-scm.com/) sollte jeder beherrschend, der Code schreibt. Sie kann aber auch hilfreich sein, wenn man zu zweit an einem tex-Dokument arbeitet und Änderungen einfach einarbeiten möchte. Ein Account bei [Github](https://github.com/) oder [Bitbucket](https://bitbucket.org/product/) ist kostenlos. Über das Intranet ist aber auch das eigene interne [Fachbereichs-Git](https://git.physik.uni-marburg.de/) erreichbar, das auf [GitLab](https://about.gitlab.com/) basiert.
### Über diesen Versuch hinaus
Wer sich intensiver mit den Möglichkeiten moderner Software auseinandersetzen möchte, sollte sich das [Missing Semester](https://git-scm.com/) anschauen.

## Sonstiges
Dieser Versuch wurde von Eric Jelli konzipiert und zuerst im Sommersemester 2019 angeboten.

## Referenzen
[1] Vicar et al. Cell segmentation methods for label-free contrast microscopy: review and comprehensive comparison.
BMC Bioinformatics, 20(1) Art. No. 360 (2019) 
