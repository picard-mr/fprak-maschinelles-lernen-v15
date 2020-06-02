# F-Praktikumsversuch 15 - Maschinelles Lernen in der wissenschaftlichen Bildanalyse
In diesem Versuch sollen Grundprinzipien des Maschinellen Lernens erlernt werden. Dabei werden Lern-Algorithmen aus dem Spektrum des Deep Learnings verwendet, um in Mikroskop-Bildern Objekte zu segmentieren. Für das Durchführen des Versuchs werden Programmierkenntnisse in Python benötigt. Während in vielen F-Praktikumsversuchen am Fachbereich ein physikalisches Phänomen untersucht wird, das durch Messungen charakterisiert wird, soll es in diesem Versuch um Vergleiche verschiedener Methoden gehen. Große physikalische Einsichten werden in diesem Versuch nicht vermittelt.

Der F-Praktikumsversuch versucht auch einige weitere Methoden / Kenntnisse zu vermitteln, mit denen Abläufe automatisiert und somit vereinfacht werden.

## Aufbau und weitere Informationen
### Begrifflichkeiten
Im Rahmen des Versuchs sollen die Studierenden folgende Begrifflichkeiten verstehen:
- U-Net as Convolutional Neuronal Network / Auto-Encoder
- semantic vs instance segmentation
- Unterscheidung von unsupervised und supervised Learning
- Loss Funktionen
- Repräsentationen von Formen / Objekten in Computern

### Paper
Dieser F-Praktikumsversuch nimmt Bezug auf verschiedene wissenschaftliche Paper:
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Olaf Ronneberger, Philipp Fischer, Thomas Brox, 2015
- Die zwei [stardist Paper](https://github.com/mpicbg-csbd/stardist) und ihre Implementierung auf Github.
- [Noise2Void - Learning Denoising from Single Noisy Images](https://arxiv.org/abs/1811.10980), Alexander Krull, Tim-Oliver Buchholz, Florian Jug, 2019 und Implementierung auf [Github](https://github.com/juglab/n2v)

### Frameworks
Im Versuch wird das Deep Learning Framework [Keras](https://keras.io/) mit Backend [tensorflow](https://www.tensorflow.org/) verwendet. Fortgeschrittene Studierende, die sich mit anderen Frameworks (bspw. [PyTorch](https://pytorch.org/)) auskennen, können den Versuch gerne auch mit anderen Frameworks durchführen.

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
