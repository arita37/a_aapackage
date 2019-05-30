# -*- coding: utf-8 -*-
import msvcrt
import os

from ctypes import windll, byref, wintypes, WinError, GetLastError
from ctypes.wintypes import HANDLE, DWORD, BOOL

# LPDWORD = POINTER(DWORD)

PIPE_NOWAIT = wintypes.DWORD(0x00000001)

ERROR_NO_DATA = 232


def pipe_no_wait(pipefd):
    """ pipefd is a integer as returned by os.pipe """

    SetNamedPipeHandleState = windll.kernel32.SetNamedPipeHandleState
    SetNamedPipeHandleState.argtypes = [HANDLE, LPDWORD, LPDWORD, LPDWORD]
    SetNamedPipeHandleState.restype = BOOL

    h = msvcrt.get_osfhandle(pipefd)

    res = windll.kernel32.SetNamedPipeHandleState(h, byref(PIPE_NOWAIT), None, None)
    if res == 0:
        print(WinError())
        return False
    return True
