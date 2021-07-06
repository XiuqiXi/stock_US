# This Python file uses the following encoding: utf-8

import pyqtgraph as pg
import numpy as np
import datetime

class alter_str_pyqt(pg.AxisItem):
    def __init__(self, xdict, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)
        self.x_values = np.asarray(xdict.keys())
        self.x_strings = xdict.values()

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            # vs is the original tick value
            vs = v * scale
            # if we have vs in our values, show the string
            # otherwise show nothing
            if vs in self.x_values:
                # Find the string with x_values closest to vs
                vstr = self.x_strings[np.abs(self.x_values-vs).argmin()]
            else:
                vstr = ""
            strings.append(vstr)
        return strings

class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(value) for value in values]

x = ['a', 'b', 'c', 'd', 'e', 'f']
y = [1, 2, 3, 4, 5, 6]

xdict = dict(enumerate(x))

stringaxis = alter_str_pyqt(xdict, orientation='bottom')

print(x)
