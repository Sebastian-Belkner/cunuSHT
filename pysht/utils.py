import time, os
from datetime import timedelta
import numpy as np
import sys
import json

class timer:
    def __init__(self, verbose, prefix='', suffix=''):
        self.t0 = time.time()
        self.ti = self.t0
        self.verbose = verbose
        self.prefix = prefix
        self.suffix = suffix
        self.keys = {}
        self.t0s = {}

    def __iadd__(self, othertimer):
        for k in othertimer.keys:
            if not k in self.keys:
                self.keys[k] = othertimer.keys[k]
            else:
                self.keys[k] += othertimer.keys[k]
        return self

    def reset(self):
        t0_, ti_ = self.t0, self.ti
        self.t0 = time.time()
        return t0_, ti_
        
    def set(self, t0, ti):
        self.t0, self.ti = t0, ti

    def reset_ti(self):
        self.ti = time.time()
        self.t0 = time.time()

    def start(self, key):
        """Starts the time tracker
        Args:
            key (_type_): _description_
        """        
        assert key not in self.t0s.keys()
        self.t0s[key] = time.time()

    def close(self, key):
        """Closes the time tracker and adds the elapsed time to the key
        Args:
            key (_type_): _description_
        """        
        assert key in self.t0s.keys()
        if key not in self.keys.keys():
            self.keys[key]  = time.time() - self.t0s[key]
        else:
            self.keys[key] += time.time() - self.t0s[key]
        del self.t0s[key]

    def __str__(self):
        if len(self.keys) == 0:
            return r""
        maxlen = np.max([len(k) for k in self.keys])
        dt_tot = time.time() - self.ti
        s = "\n"
        s += "  " + 27*('--') + '\n'
        ts = "\r | {0:%s}" % (str(maxlen) + "s")
        for k in self.keys:
            _ = str(timedelta(seconds=self.keys[k], ))
            _ = ':'.join(str(_).split(':')[2:])
            s += ts.format(k) + ":  [" + _ + "] " + "(%.1f%%)  \t|\n"%(100 * self.keys[k]/dt_tot)
        _ = str(timedelta(seconds=dt_tot))
        _ = ':'.join(str(_).split(':')[2:])
        s += "  " + 27*('- ') + '\n'
        s += ts.format("Total") + ":  [" + _ + "] " + "sec.mus  \t|"
        s += "\n  " + 27*('--') + '\n'
        return s

    def add(self, label):
        """Add the elapsed time since the last call to the label

        Args:
            label (_type_): _description_
        """        
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time.time()
        self.keys[label] += t0 - self.t0
        self.t0 = t0

    def add_elapsed(self, label):
        """Add the elapsed time since the start of the timer to the label

        Args:
            label (_type_): _description_
        """        
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time.time()
        self.keys[label] += t0 - self.ti

    def dumpjson(self, fn):
        """Dump the timer keys to a json file

        Args:
            fn (function): _description_
        """        
        json.dump(self.keys, open(fn, 'w'))

    def checkpoint(self, msg):
        """Prints a message and the time elapsed since the last checkpoint

        Args:
            msg (_type_): _description_
        """        
        dt = time.time() - self.t0
        self.t0 = time.time()

        if self.verbose:
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            dhi = np.floor((self.t0 - self.ti) / 3600.)
            dmi = np.floor(np.mod((self.t0 - self.ti), 3600.) / 60.)
            dsi = np.floor(np.mod((self.t0 - self.ti), 60))
            sys.stdout.write("\r  %s   [" % self.prefix + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] "
                             + " (total [" + (
                                 '%02d:%02d:%02d' % (dhi, dmi, dsi)) + "]) " + msg + ' %s \n' % self.suffix)