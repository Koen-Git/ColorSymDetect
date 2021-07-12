Find rotational and reflection symmetries inside images in a given folder
Uses https://github.com/mawady/ColorSymDetect to fetch reflection symmetries
Usage:
python main.py
arguements:
--input "custom input folder" (default: ./input/)
--output "custom output folder" (default: ./output/)
--mode "slow/fast, either uses Machine learning (slow) to detect rotational symmetries or simple rules (fast)" (default: slow)
Requires matlab to be installed with the python extension
Tested on python 3.7.10 with matlab R2020B

Based on the following paper:

-   Elawady, Mohamed, Christophe Ducottet, Olivier Alata, Cécile Barat, and Philippe Colantoni. "Wavelet-based reflection symmetry detection via textural and color histograms." In Proceedings, ICCV Workshop on Detecting Symmetry in the Wild, Venice, vol. 3, p. 7. 2017.

```

```
