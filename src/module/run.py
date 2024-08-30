#!/usr/bin/env python3
import os
import ctypes
from savant.entrypoint.main import main

if __name__ == '__main__':
    ctypes.cdll.LoadLibrary('/opt/savant/src/libmmdeploy_tensorrt_ops.so')
    main(os.path.join(os.path.dirname(__file__), 'module.yml'))
    print('done')
