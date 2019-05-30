# -*- coding: utf-8 -*-
"""Automatic release tools."""

from __future__ import division, print_function

import copy
import datetime
import gc
#####################################################################################################
import os
import os.path as op
import re
import shutil
import sys
import time
from builtins import map, next, object, range, str, zip
from subprocess import call

import IPython
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import pandas as pd
import scipy as sci
import six
import urllib3
from bs4 import BeautifulSoup
from future import standard_library
from past.builtins import basestring
from past.utils import old_div
from six.moves import input

import arrow
from github3 import login

standard_library.install_aliases()








# CFG   = {'plat': sys.platform[:3]+"-"+os.path.expanduser('~').split("\\")[-1].split("/")[-1], "ver": sys.version_info.major}
# DIRCWD= {'win-asus1': 'D:/_devs/Python01/project27/', 'win-unerry': 'G:/_devs/project27/' , 'lin-noel': '/home/noel/project27/', 'lin-ubuntu': '/home/ubuntu/project27/', 'lin-travis': '/home/ubuntu/project27/'}[CFG['plat']]
# print(os.environ)
# DIRCWD= os.environ["DIRCWD"]; os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')

# __path__= DIRCWD +'/aapackage/'
__version__ = "1.0.0"
# __file__= "util.py"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _call(cmd, system=False):
    if system:
        ret = os.system(cmd)
    else:
        ret = call(cmd.split(" "))
    if ret != 0:
        raise RuntimeError()


# -----------------------------------------------------------------------------
# Messing with version in __init__.py
# -----------------------------------------------------------------------------

root = op.realpath(op.join(op.dirname(__file__), "../"))
_version_pattern = r"__version__ = '([0-9\.]+)((?:\.dev)?)([0-9]+)'"
_version_replace = r"__version__ = '{}{}{}'"


def _path(fn):
    return op.realpath(op.join(root, fn))


def _get_stable_version():
    fn = _path("phy/__init__.py")
    with open(fn, "r") as f:
        contents = f.read()
    m = re.search(_version_pattern, contents)
    return m.group(1)


def _update_version(dev_n="+1", dev=True):
    fn = _path("phy/__init__.py")
    dev = ".dev" if dev else ""

    def func(m):
        if dev:
            if isinstance(dev_n, six.string_types):
                n = int(m.group(3)) + int(dev_n)
            assert n >= 0
        else:
            n = ""
        if not m.group(2):
            raise ValueError()
        return _version_replace.format(m.group(1), dev, n)

    with open(fn, "r") as f:
        contents = f.read()

    contents_new = re.sub(_version_pattern, func, contents)

    with open(fn, "w") as f:
        f.write(contents_new)


def _increment_dev_version():
    _update_version("+1")


def _decrement_dev_version():
    _update_version("-1")


def _set_final_version():
    _update_version(dev=False)


# -----------------------------------------------------------------------------
# Git[hub] tools
# -----------------------------------------------------------------------------


def _create_gh_release():
    version = _get_stable_version()
    name = "Version {}".format(version)
    path = _path("dist/phy-{}.zip".format(version))
    assert op.exists(path)

    with open(_path(".github_credentials"), "r") as f:
        user, pwd = f.read().strip().split(":")
    gh = login(user, pwd)
    phy = gh.repository("kwikteam", "phy")

    if input("About to create a GitHub release: are you sure?") != "yes":
        return
    release = phy.create_release(
        "v" + version,
        name=name,
        # draft=False,
        # prerelease=False,
    )

    release.upload_asset("application/zip", op.basename(path), path)


def _git_commit(message, push=False):
    assert message
    if input("About to git commit {}: are you sure?") != "yes":
        return
    _call('git commit -am "{}"'.format(message))
    if push:
        if input("About to git push upstream master: are you sure?") != "yes":
            return
        _call("git push upstream master")


# -----------------------------------------------------------------------------
# PyPI
# -----------------------------------------------------------------------------


def _upload_pypi():
    _call("python setup.py sdist --formats=zip upload")


# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------


def _build_docker():
    _call("docker build -t phy-release-test docker/stable")


def _test_docker():
    _call(
        "docker run --rm phy-release-test /sbin/start-stop-daemon --start "
        "--quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile "
        "--background --exec /usr/bin/Xvfb -- :99 -screen 0 1400x900x24 "
        "-ac +extension GLX +render && "
        'python -c "import phy; phy.test()"',
        system=True,
    )


# -----------------------------------------------------------------------------
# Release functions
# -----------------------------------------------------------------------------


def release_test():
    _increment_dev_version()
    _upload_pypi()
    _build_docker()
    _test_docker()


def release():
    version = _get_stable_version()
    _set_final_version()
    _upload_pypi()
    _git_commit("Release {}.".format(version), push=True)
    _create_gh_release()


if __name__ == "__main__":
    globals()[sys.argv[1]]()  ### Execute the function    python release.py release using global var
