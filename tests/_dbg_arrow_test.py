import numpy as np
import pytest
from tests.test_geo2d_registration import *  # noqa
from tests.test_geo2d_registration import _geo2d_recording  # noqa


def test_dbg(dialog, qt_app):
    from PySide6.QtCore import QPoint, Qt, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication
    dialog.show(); QApplication.processEvents()
    canvas = dialog.probe_canvas
    canvas.draw()
    ax = dialog.probe_axes
    tail = ax.transData.inverted().transform(ax.transAxes.transform((0.3, 0.3)))
    dialog.start_direction(tail)
    print("mouseTracking", canvas.hasMouseTracking())
    for frac in (0.5, 0.6, 0.7):
        px, py = ax.transAxes.transform((frac, frac))
        h = canvas.figure.bbox.height / canvas.device_pixel_ratio
        pos = QPointF(px / canvas.device_pixel_ratio, h - py / canvas.device_pixel_ratio)
        ev = QMouseEvent(QEvent.MouseMove, pos, canvas.mapToGlobal(pos), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
        QApplication.sendEvent(canvas, ev)
        QApplication.processEvents()
        print(frac, dialog._arrow and dialog._arrow._posA_posB, dialog._arrow in ax.patches if dialog._arrow else None, dialog._arrow.get_visible() if dialog._arrow else None)
