"""
Sample radar files
"""

import os
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

DUALPOL_SCAN = os.path.join(DATA_PATH, 'example_dualpol_scan')
LEGACY_SCAN = os.path.join(DATA_PATH, 'example_legacy_scan')

CANADA_IRIS_DOPVOL1_A = os.path.join(DATA_PATH, 'CANADA_IRIS_DOPVOL1_A.iri')
CANADA_IRIS_DOPVOL1_B = os.path.join(DATA_PATH, 'CANADA_IRIS_DOPVOL1_B.iri')
CANADA_IRIS_DOPVOL1_C = os.path.join(DATA_PATH, 'CANADA_IRIS_DOPVOL1_C.iri')

CANADA_IRIS_DOPVOL_FILES = [CANADA_IRIS_DOPVOL1_A, 
                            CANADA_IRIS_DOPVOL1_B, 
                            CANADA_IRIS_DOPVOL1_C]