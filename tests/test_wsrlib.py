import pytest
import numpy as np

'''Tests for aws_key'''
from wsrlib import aws_key

def test_aws_key_generates_correct_string_1():
    name = 'KBGM20170421_025222_V06'
    assert aws_key(name) == '2017/04/21/KBGM/KBGM20170421_025222_V06'

def test_aws_key_generates_correct_string_2():
    name = 'KBGM20170421_025222'
    assert aws_key(name) == '2017/04/21/KBGM/KBGM20170421_025222'

def test_aws_key_generates_correct_string_3():
    name = 'KBGM20170421_025222_V04.gz'
    assert aws_key(name) == '2017/04/21/KBGM/KBGM20170421_025222_V04.gz'
    
def test_aws_key_error_on_bad_key_1():
    with pytest.raises(ValueError):
        aws_key('KBGM20170421_02522')

def test_aws_key_error_on_bad_key_2():
    with pytest.raises(ValueError):
        aws_key('123420170421_02522')

def test_aws_key_error_on_bad_key_3():
    with pytest.raises(ValueError):
        aws_key('foo')

        
'''Tests for prefix2key'''
from wsrlib import prefix2key

def test_prefix2key_generate_correct_string_1():
    bucket = 'noaa-nexrad-level2'
    key = '2016/01/07/KBUF/KBUF20160107_121946'
    assert prefix2key(bucket, key) == '2016/01/07/KBUF/KBUF20160107_121946_V06.gz'

def test_prefix2key_generate_correct_string_2():
    bucket = 'noaa-nexrad-level2'
    key = '2016/01/07/KBUF/KBUF20160107_1219'
    assert prefix2key(bucket, key) == '2016/01/07/KBUF/KBUF20160107_121946_V06.gz'

def test_prefix2key_generate_correct_string_with_multiple_matches():
    bucket = 'noaa-nexrad-level2'
    key = '2016/01/07/KBUF/KBUF20160107_12'
    assert prefix2key(bucket, key) == '2016/01/07/KBUF/KBUF20160107_120000_V06.gz'
    
def test_prefix2key_raises_key_error():
    bucket = 'noaa-nexrad-level2'
    key = 'KBUF20160107_121946'
    with pytest.raises(KeyError):
        prefix2key(bucket, key)

        
'''Tests for idb and db'''
from wsrlib import idb, db

def test_idb_correct_values():
    x = np.arange(11)
    refvals = np.array([1.000000000000000,
                        1.258925411794167,
                        1.584893192461114,
                        1.995262314968880,
                        2.511886431509580,
                        3.162277660168380,
                        3.981071705534972,
                        5.011872336272722,
                        6.309573444801933,
                        7.943282347242816,
                        10.000000000000000
                    ])
    assert np.allclose(idb(x), refvals)

    
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_db_correct_values():
    x = np.arange(0, 550, 50)
    
    refvals = [-np.inf,
              16.989700043360187,
              20.000000000000000,
              21.760912590556813,
              23.010299956639813,
              23.979400086720375,
              24.771212547196626,
              25.440680443502757,
              26.020599913279625,
              26.532125137753436,
              26.989700043360187]
    
    assert np.allclose(db(x), refvals)

def test_idb_then_db_is_identity():
    dbz = np.linspace(-15, 70, 100)
    assert np.allclose(db(idb(dbz)), dbz)


'''z to reflectivity and back conversions'''
from wsrlib import idb, z_to_refl, refl_to_z

def test_z_to_refl_correct_values():

    refvals = 1e8 * np.array([
        0.000000006840289,
        0.000000060189242,
        0.000000529618708,
        0.000004660234407,
        0.000041006453134,
        0.000360825025444,
        0.003174980741723,
        0.027937371300401,
        0.245827228152834,
        2.163089198747852])
        
    z = idb(np.linspace(-15, 70, 10))
    eta, _ = z_to_refl(z)
    
    assert np.allclose(eta, refvals)


def test_z_to_refl_to_z_is_identity():
    z = idb(np.linspace(-15, 70, 100))
    eta, _ = z_to_refl(z)
    z2, _ = refl_to_z(eta)
    assert np.allclose(z, z2)


'''Read a file from s3'''
import tempfile
import os
from wsrlib import get_s3, read_s3

def test_get_s3():
    name = 'KBUF20160107_121946_V06'
    with tempfile.NamedTemporaryFile() as temp:
        get_s3(name, temp.name)
        assert os.stat(temp.name).st_size > 0

def test_read_s3():
    name = 'KBUF20160107_121946_V06'
    radar = read_s3(name)
    refvals = np.array([2375., 2625., 2875., 3125., 3375., 3625., 3875., 4125., 4375.])
    assert np.allclose(radar.range['data'][1:10], refvals)


'''Sweep selection'''
from wsrlib import get_sweeps
from wsrlib.testing import LEGACY_SCAN, DUALPOL_SCAN
from pyart.io import read_nexrad_archive

def test_get_sweeps():
    radar = read_nexrad_archive(DUALPOL_SCAN)
    
    for field in ['reflectivity', 
                  'differential_reflectivity', 
                  'cross_correlation_ratio',
                  'differential_phase']:
        sweeps = get_sweeps(radar, field)
        sweepnums = [sweep['sweepnum'] for sweep in sweeps]
        assert np.array_equal(sweepnums, [0, 2, 4, 6, 7])

    for field in ['velocity', 
                  'spectrum_width']:
        sweeps = get_sweeps(radar, field)
        sweepnums = [sweep['sweepnum'] for sweep in sweeps]
        assert np.array_equal(sweepnums, [1, 3, 5, 6, 7])


'''radar2mat errors / warnings'''
from wsrlib import radar2mat
import warnings

def test_radar2mat_raises_value_error_for_bad_field():
    radar = read_nexrad_archive(DUALPOL_SCAN)
    with pytest.raises(ValueError):
        f = radar2mat(radar, fields=['reflectivity', 'foo'])

def test_radar2mat_warns_if_field_unavailable():
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        radar = read_nexrad_archive(LEGACY_SCAN)
    
    with pytest.warns(UserWarning):
        f = radar2mat(radar, fields=['differential_reflectivity', 'velocity'])

def test_radar2mat_value_error_if_bad_coords():
    radar = read_nexrad_archive(DUALPOL_SCAN)
    with pytest.raises(ValueError):
        f = radar2mat(radar, coords='foo')
