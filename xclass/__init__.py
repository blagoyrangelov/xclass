"""xclass — Extragalactic X-ray Source Classifier.

ML pipeline to classify Chandra X-ray point sources in nearby spiral galaxies
into 7 source classes using SED-translated HST photometry and a two-stage
Random Forest classifier.

Source classes: AGN, LMXB, HMXB, CV, LM-STAR, HM-STAR, SNR
Target galaxies: M31, M33, M51, M83, NGC 6946
"""

__version__ = "0.1.0"
__author__ = "xclass team"
