# F-Praktikumsversuch 15 - Maschinelles Lernen in der wissenschaftlichen Bildanalyse
In diesem Versuch sollen Grundprinzipien der Bildsegmentierung von Mikroskopbildern mithilfe von *Deep Learning* vermittelt werden. Dazu werden Intensitäten-basierte Verfahren mit den Verfahren auf Grundlage von neuronalen Netzwerken verglichen. Neuronale Netzwerke bilden eine Klasse von Lern-Algorithmen welche auch unter schlechten Signal-zu-Rausch-Verhältnissen gute Segmentierungen poduzieren können [[1](https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-019-2880-8)]. Die Segmentierung von Mikroskopbildern ist eine Grundvoraussetzung um physikalische Phänomene innerhalb von bakteriellen Kolonien quantitativ zu beschreiben. Die Kenntnisse über verschiedene Segmentierungsverfahren gehören zum Grundlagen der Bildanalyse und sind unverzichtbar in der Auswertung von Mikroskopieexperimenten. Für diesen Versuch werden werden Programmierkenntnisse in Python benötigt.

Neben theoretischen Kenntnissen sollen auch die praktischen Fertigkeiten bei der Erstellung von wissenschaftlicher Software geschult werden. Dazu werden einige kleinere Werkzeuge/ Methoden vorgestellt, die das Programmieren erleichtern bzw. wiederkehrende Aufgaben automatisieren können und sich in der Praxis bewährt haben.

[1] Cell segmentation methods for label-free contrast microscopy: review and comprehensive comparison. Tomas Vicar, Jan Balvan, Josef Jaros, Florian Jug, Radim Kolar, Michal Masarik, Jaromir Gumulec.
BMC Bioinformatics, 20(1) Art. No. 360 (2019) 

## Aufbau und weitere Informationen
### Begrifflichkeiten
Für einen reibunglosen Ablauf werden Kenntnisse über folgende Begrifflichkeiten vorausgesetzt:
- *U-Net* als Beispiel eines *Convolutional Neuronal Network* / *Auto-Encoder*
- *Semantical* vs. *Instance Segmentation*
- Unterscheidung von *Unsupervised* und *Supervised Learning*
- *Loss Function*
- Repräsentationen von Formen / Objekten in Computern

### Literatur
Die in verwendeten Algorithmen wurden in folgenden Publikationen vorgestellt:
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28). Olaf Ronneberger, Philipp Fischer, Thomas Brox.
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015 ([Preprint Link](https://arxiv.org/abs/1505.04597))
- [Cell Detection with Star-convex Polygons](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_30).
Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.11071, 2018 ([Preprint Link](https://arxiv.org/abs/1806.03535))
- [Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy](https://ieeexplore.ieee.org/document/9093435). Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers. IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, CO, USA, 2020, pp. 3655-3662 ([Open Access Version](http://openaccess.thecvf.com/content_WACV_2020/papers/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.pdf)) ([GitHub](https://github.com/mpicbg-csbd/stardist))
- [Noise2Void - Learning Denoising from Single Noisy Images](https://ieeexplore.ieee.org/abstract/document/8954066). Alexander Krull, Tim-Oliver Buchholz, Florian Jug.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 2124-2132 ([Open Access Version](http://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)) ([GitHub](https://github.com/juglab/n2v))

Im Rahmen des Versuchs wird mit den auf GitHub zur Verfügung gestellten Python Modulen gearbeitet.

### Frameworks
Um die Benutztung von neuronalen Netzwerken zu erleichtern, können Studierende mittlerweile auf zahlreichen Deep Learning Framework zurückgreifen. Diese übernehmen zumeist die hardware-nahe Implementierung der rechenlastigen Algorithmen, erleichtern die Nutzung von Graphic Processing Units (GPUs) zur beschleunigten Ausführung und erlauben den Zugriff über eine komfortable Python Schnittstelle. Der Versuch ist auf die Benutzung von [Keras](https://keras.io/) mit [tensorflow](https://www.tensorflow.org/) Backend ausgelegt. Fortgeschrittene Studierende, die Erfahrungen mit anderen Frameworks mitbringen, ist die Frameworkwahl freigestellt.

### Ablauf
Für die Berarbeitung der Aufgaben wird ein Computer mit Internetzugang und einem aktuellen Browser benötigt - auf Anfrage kann dieser auch gestellt werde. Alle Aufgaben werden in Form von [Jupyter Notebooks](https://jupyter.org/) gestellt und sind in diesem GitHub-Repository zu finden. Um die Aufgaben innerhalb der gegebenen Zeit zu berarbeiten wird der Zugriff auf aktuelle GPUs benötigt. Zugriff auf entsprechende Hardware kann die Student-Workstation [Picard](https://picard.physik.uni-marburg.de) zur Verfügung gestellt werden. Mit [diesem Link](https://picard.physik.uni-marburg.de/jupyterhub/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fpicard-mr%2Ffprak-maschinelles-lernen-v15&urlpath=tree%2Ffprak-maschinelles-lernen-v15%2F) werden die Aufgaben (inkl. dieser Anleitung) in das entsprechende Verzeichnis kopiert und steht unmittelbar zur Bearbeitung zur Verfügung. Unabhängig von der Workstation steht es den Studierenden frei die Aufaben auch auf eigener Hardware zu lösen.

Die Notebooks sind nummeriert. Es wird empfohlen die Aufgaben in der vorgeschlagenen Reihenfolge zu bearbeiten:
1. **Die Daten:** Macht euch mit den Daten vertraut. Implementierung eines Viewers. Vorbereiten der Daten.
5. **Otsu-Thresholding / Kantenfinden:** Ausprobieren klassischer Filter-Methoden.
10. **Entrauschen:** *Noise2Void* Training.
20. **Instance Segmentation:** *Stardist* Training.
30. **Implementierung U-Net / ResNet:** Eigene Implementierung zur *Semantic Segmentation*. Anschließendes Training.
40. **Validierung:** Die trainierten Modelle sollen validiert werden.

Obwohl die Aufgaben an einem beliebigen Ort mit Internetzugang bearbeitet werden können, raten wir ausrücklich zu einer Bearbeitung vor Ort. Erfahrungsgemäß lassen sich Probleme und Fragen schneller persönlich aus dem Weg räumen.

Sollte dennoch eine Bearbeitung Zuhause/ remote gewünscht sein, wird um eine kurze Rückmeldung gebeten. Bitte teilt auch zeitnah mit, ob ihr einen Computer gestellt bekommen wollt.

### Aufbau des Versuchsprotokolls
Generell sollten alle ausgeführten Aufgaben protokolliert werden und die verschiedenen Methoden miteinander verglichen werden. Auf einen Grundlagenteil, der die verwendeten Methoden erläutert, wird Wert gelegt.

## Werkzeuge/ Tools
* [Project Juypter](https://jupyter.org/)
* [Git](https://git-scm.com/) (Versionskontrolle) z.B. im Webinterface: [Fachbereich Physik](https://git.physik.uni-marburg.de/) (VPN), [GitHub](https://github.com/), [Bitbucket](https://bitbucket.org/product/), [GitLab](https://about.gitlab.com/). (Auch sehr nützlich für tex-Dokumente)
* Für eine Übersicht über weitere nützliche Werkzeuge/ Methoden in der wissenschaftlichen Softwareentwicklung ist das [Missing Semester](https://missing.csail.mit.edu/) sehr zu empfehlen.

## Sonstiges
Dieser Versuch wurde von Eric Jelli konzipiert und zuerst im Sommersemester 2019 angeboten.
