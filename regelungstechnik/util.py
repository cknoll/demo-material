# based on https://stackoverflow.com/questions/37512502/how-to-make-arrow-that-loops-in-matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
import numpy as np
from numpy import radians as rad


def drawCirc(ax, radius, centX, centY, startangle, diffangle, **kwargs):

    startarrow = kwargs.pop("startarrow", False)
    endarrow = kwargs.pop("endarrow", True)
    if diffangle < 0:

        startangle, diffangle = (startangle + diffangle) % 360, -diffangle
        startarrow, endarrow = endarrow, startarrow

    defaults = {
        "linestyle": "-",
        "lw": 2,
        "color": "black",
        "k": 0.05,  # polygon size as fraction of radius
    }
    defaults.update(kwargs)
    kwargs = defaults

    k = kwargs.pop("k")

    # ========Line
    theta2 = startangle + diffangle
    arc = Arc(
        [centX, centY],
        radius,
        radius,
        angle=0,
        theta1=startangle,
        theta2=theta2,
        capstyle="round",
        **kwargs
    )
    ax.add_patch(arc)

    # ========Create the arrow heads

    if startarrow:
        startX = centX + (radius / 2) * np.cos(rad(startangle))  # Do trig to determine end position
        startY = centY + (radius / 2) * np.sin(rad(startangle))

        ax.add_patch(  # Create triangle as arrow head
            RegularPolygon(
                (startX, startY),  # (x,y)
                3,  # number of vertices
                radius * k,  # radius
                rad(startangle + 180),  # orientation
                color=kwargs["color"],
            )
        )

    if endarrow:
        endX = centX + (radius / 2) * np.cos(rad(theta2))  # Do trig to determine end position
        endY = centY + (radius / 2) * np.sin(rad(theta2))

        ax.add_patch(  # Create triangle as arrow head
            RegularPolygon(
                (endX, endY),  # (x,y)
                3,  # number of vertices
                radius * k,  # radius
                rad(theta2),  # orientation
                color=kwargs["color"],
            )
        )
