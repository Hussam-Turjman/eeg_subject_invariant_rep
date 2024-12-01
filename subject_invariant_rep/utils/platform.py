from __future__ import annotations
# This file was adapted from gn/build/gen.py  https://gn.googlesource.com/gn/

# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.


import sys

import os
import platform as p


class Platform(object):
    """Represents a host/target platform."""

    def __init__(self, platform=None):
        self._platform = platform
        if self._platform is not None:
            return
        self._platform = sys.platform
        if self._platform.startswith('linux'):
            self._platform = 'linux'
        elif self._platform.startswith('darwin'):
            self._platform = 'darwin'
        elif self._platform.startswith('mingw'):
            self._platform = 'mingw'
        elif self._platform.startswith('msys'):
            self._platform = 'msys'
        elif self._platform.startswith('win'):
            self._platform = 'msvc'
        elif self._platform.startswith('aix'):
            self._platform = 'aix'
        elif self._platform.startswith('fuchsia'):
            self._platform = 'fuchsia'
        elif self._platform.startswith('freebsd'):
            self._platform = 'freebsd'
        elif self._platform.startswith('netbsd'):
            self._platform = 'netbsd'
        elif self._platform.startswith('openbsd'):
            self._platform = 'openbsd'
        elif self._platform.startswith('haiku'):
            self._platform = 'haiku'
        elif self._platform.startswith('sunos'):
            self._platform = 'solaris'

    @staticmethod
    def known_platforms():
        return ['linux', 'darwin', 'mingw', 'msys',
                'msvc', 'aix', 'fuchsia',
                'freebsd', 'netbsd', 'openbsd', 'haiku',
                'solaris']

    def platform(self):
        return self._platform

    def is_linux(self):
        return self._platform == 'linux'

    def is_mingw(self):
        return self._platform == 'mingw'

    def is_msys(self):
        return self._platform == 'msys'

    def is_msvc(self):
        return self._platform == 'msvc'

    def is_windows(self):
        return self.is_mingw() or self.is_msvc()

    def is_darwin(self):
        return self._platform == 'darwin'

    def is_aix(self):
        return self._platform == 'aix'

    def is_haiku(self):
        return self._platform == 'haiku'

    def is_solaris(self):
        return self._platform == 'solaris'

    def is_posix(self):
        return self._platform in ['linux', 'freebsd',
                                  'darwin', 'aix', 'openbsd',
                                  'haiku', 'solaris',
                                  'msys', 'netbsd']


PLATFORM = Platform(None)

IS_LINUX = PLATFORM.is_linux()
IS_WIN = PLATFORM.is_windows()
OS_NAME = os.name.lower()
SYS_ARCH = p.architecture()

__all__ = ["PLATFORM", "IS_WIN", "IS_LINUX", "SYS_ARCH", "OS_NAME"]