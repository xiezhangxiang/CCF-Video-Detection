# coding=utf-8
import ctypes as _ctp
import os as _os
import sys as _sys
import numpy as _np
import cv2 as _cv2

__version__ = '0.1.1b'
__all__ = ['CdvsError', ]

if _sys.version[0] == '2':
    def _s2b(strval):
        if isinstance(strval, unicode):
            return strval.encode()
        else:
            return strval
else:
    def _s2b(strval):
        if isinstance(strval, bytes):
            return strval
        else:
            return strval.encode()

_prefix = _os.path.split(_os.path.realpath(__file__))[0]
_os.environ['PATH'] = ';'.join([_os.environ["PATH"], _prefix])


if _os.name in ("nt", "ce"):
    _dll = _ctp.CDLL(_prefix + '\\boyun_search.dll')
    # convert from 'bytes'(str) to str
    if _sys.version[0] == '2':
        def _scvt(s):
            return s.decode('gb2312').encode('utf-8')
    else:
        def _scvt(s):
            return s.decode('gb2312')
else:
    _dll = _ctp.CDLL(_prefix + '/libboyun_search.so')
    # convert from 'bytes'(str) to str
    if _sys.version[0] == '2':
        def _scvt(s):
            return s
    else:
        def _scvt(s):
            return s.decode()


class _bys_buffer(_ctp.Structure):
    _fields_ = [('data', _ctp.c_void_p), ('size', _ctp.c_size_t)]


class _bys_image(_ctp.Structure):
    _fields_ = [('pixel_data', _ctp.c_void_p), ('width', _ctp.c_int32), ('height', _ctp.c_int32),
                ('widthStep', _ctp.c_int32), ('type_flag', _ctp.c_uint32)]


class _bys_indexSearchResult(_ctp.Structure):
    _fields_ = [('hrec', _ctp.c_void_p), ('similarity', _ctp.c_float)]


class CdvsError(Exception):
    """
        exception class for 'cdvs' sdk

        Attributes:
            errc: err-code (int)
            errs: err-message (string)
    """
    __slots__ = ('errc', 'errs')

    def __init__(self, errc, errs):
        self.errc = errc
        self.errs = errs

    def __str__(self):
        return repr(self.errc) + ',' + self.errs

    def __repr__(self):
        return 'CdvsError:' + self.__str__()


_dll.BYS_globalInit.restype = _ctp.c_int32
_dll.BYS_set_remote_addr.restype = _ctp.c_int32
_dll.BYS_getFeature_Cuda.restype = _ctp.c_int32
_dll.BYS_featureMatchScore.restype = _ctp.c_int32
_dll.BYS_newIndex.restype = _ctp.c_void_p


def _im2_bys_img(im):
    # type: (_np.ndarray) -> _bys_image
    assert type(im) == _np.ndarray
    bysimg = _bys_image()

    if im.strides[1] == 3:
        im = _cv2.cvtColor(im, _cv2.COLOR_BGR2GRAY)
    assert im.strides[1] == 1
    bysimg.pixel_data = im.ctypes._data
    shp = im.shape
    bysimg.width, bysimg.height = _ctp.c_int32(shp[1]), _ctp.c_int32(shp[0])
    bysimg.widthStep, bysimg.type_flag = _ctp.c_int32(im.strides[0]), _ctp.c_uint32(1)
    return bysimg


'''========================global funtions============================='''


def global_init():
    rt = _dll.BYS_globalInit()
    if rt != 0:
        raise CdvsError(-1, "BYS_globalInit failed: {}".format(rt))


'''====================remote cuda functions====================='''


def set_remote_addr(ip, port):
    assert isinstance(ip,str) and isinstance(port,int)
    _dll.BYS_set_remote_addr(_s2b(ip), _ctp.c_int32(port))


class Feature(object):
    __slots__ = ('_hf', )
    __dl = _dll

    def __init__(self, p):
        assert isinstance(p, _ctp.c_void_p)
        self._hf = p

    def __del__(self):
        self.__dl.BYS_releaseFeature(self._hf)

    def __repr__(self):
        return "cdvs.Feature@" + hex(self._hf.value)


def getfeature_cuda(im):
    # type: (_np.ndarray) -> Feature
    hfeat = _ctp.c_void_p(0)
    bimg = _im2_bys_img(im)
    errbuf = _ctp.create_string_buffer(256)
    rt = _dll.BYS_getFeature_Cuda(_ctp.byref(bimg), _ctp.c_int32(13), _ctp.byref(hfeat), errbuf)
    if rt != 0:
        raise CdvsError(rt, _ctp. errbuf.value.decode())
    return Feature(hfeat)


def getfeature_cpu(im):
    # type: (_np.ndarray) -> Feature
    hfeat = _ctp.c_void_p(0)
    bimg = _im2_bys_img(im)
    rt = _dll.BYS_getFeature(_ctp.byref(bimg), _ctp.c_int32(13), _ctp.byref(hfeat))
    if rt != 0:
        raise CdvsError(rt, 'BYS_getFeature failed: {}'.format(rt))
    return Feature(hfeat)


'''def release_feature(hfeat):
    assert isinstance(hfeat, _ctp.c_void_p)
    _dll.BYS_releaseFeature(hfeat)'''


def feature_match_score(f1, f2, global_sim=False):
    # type: (Feature, Feature, bool) -> float
    score = _ctp.c_float()
    ftype = _ctp.c_int32(0 if global_sim else 13)
    rt = _dll.BYS_featureMatchScore(f1._hf, f2._hf, ftype, _ctp.byref(score))
    if rt != 0:
        raise CdvsError(rt, 'BYS_featureMatchScore failed: {}'.format(rt))
    return float(score.value)


class IndexDB(object):
    __slots__ = ('__p', )
    __dl = _dll

    def __init__(self):
        self.__p = _ctp.c_void_p(self.__dl.BYS_newIndex())

    def __del__(self):
        self.__dl.BYS_releaseIndex(self.__p)

    def record_count(self):
        return int(self.__dl.BYS_indexGetRecCount(self.__p))

    def clear_records(self):
        self.__dl.BYS_indexClearRecords(self.__p)

    def add_record(self, feat, bind_data):
        # type: (Feature, str) -> None
        assert isinstance(feat, Feature) and isinstance(bind_data, str)
        bf = _bys_buffer()
        bind_data = _s2b(bind_data)
        bf.data = _ctp.cast(bind_data, _ctp.c_void_p)
        bf.size = _ctp.c_size_t(len(bind_data))
        rt = _dll.BYS_indexAddRecord(self.__p, feat._hf, _ctp.byref(bf), _ctp.c_int32(1))
        if rt != 0:
            raise CdvsError(rt, 'BYS_indexAddRecord failed: {}'.format(rt))

    def retrieve2(self, feat, nret, min_sim, sel1, s1_thres):
        # type: (Feature, int, float, int, float) -> list
        assert isinstance(feat, Feature) and isinstance(nret, int) and isinstance(min_sim, float) and isinstance(sel1, int)
        res = (_bys_indexSearchResult * nret)()
        rct = _ctp.c_int32(nret)
        rt = _dll.BYS_indexRetrieve2(self.__p, feat._hf, _ctp.c_float(min_sim), _ctp.c_int32(sel1),
                                     _ctp.c_float(s1_thres), res, _ctp.byref(rct))
        if rt != 0:
            raise CdvsError(rt, "BYS_indexRetrieve2 failed: {}".format(rt))
        lst = []
        for i in range(int(rct.value)):
            hrec = _ctp.c_void_p(res[i].hrec)
            buf = _bys_buffer()
            _dll.BYS_indexGetBindData_buffer(hrec, 0, _ctp.byref(buf))
            bd = _ctp.string_at(buf.data, buf.size)
            _dll.BYS_indexRecordDeref(hrec)
            lst.append((float(res[i].similarity), bd.decode()))
        return lst


global_init()

'''#---Example---#

import cfldwp as cfl

with open("mod.cfg","r") as f:
    json_data = f.read()

cfl.load_mod_cfg_json(json_data)
# You can do something others...
cfl.wait_init_done()



det = cfl.mod("detection-net")
print(det.num_inputs(), det.num_outputs())

out_data = det.sync_proc([(0,1,3,406,406,nd.ctypes.data)], [0,1,"fc7"])
print(out_data[0], out_data[1], out_data[2])
'''
