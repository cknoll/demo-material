# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# import numtools


"""
This script creates an animation with a translated and rotated rectangle.

useful links:

https://matplotlib.org/api/animation_api.html

https://matplotlib.org/users/transforms_tutorial.html
https://matplotlib.org/api/transformations.html

"""


fig, ax = plt.subplots()
ln, = plt.plot([], [], 'r-', animated=True)


# False -> faster (non-persistent)
# True -> slower but save images
save_images = False


N = 100  # number of frames (or time steps: more frames -> slower animation)
# array with time values
tt = np.linspace(0, 1, N)

# load geometrical information about the path
nodes = np.load("nodes.npy")
spline_tuple, u = interpolate.splprep( nodes.T, s=0 )

XX, YY = interpolate.splev( tt, spline_tuple, der=0 )
dXX, dYY = interpolate.splev( tt, spline_tuple, der=1 )

PHI = np.arctan2(dYY, dXX)

# handle possible discontinuities
#PHI = numtools.cont_continuation(PHI, 2*np.pi, 0.5*np.pi)


# plot the path as green dashed line
plt.plot(XX, YY, '--g')

ax = plt.gca()


# width and height for the rectangle
box_w = .1
box_h = .05

# Meaning of arguments: (corner_x, corner_y), width, height,
# (color_R, color_G, color_B, color_alpha)
rect1 = patches.Rectangle((-box_w*.1, -box_h*.5), box_w, box_h,
                          fc=(.4, .4, .8, .5))


if 0:
    # this might be useful for debugging
    ax.axis('equal')

    # rot = mpl.transforms.Affine2D().translate(XX[0], YY[0]) + ax.transData
    i = 0
    rot = mpl.transforms.Affine2D().rotate(PHI[i]).translate(XX[i], YY[i])
    rot += ax.transData
    rect1.set_transform(rot)

    ax.add_patch(rect1)
    plt.grid()

    # visualization of x(t), y(t), phi(t)
    plt.figure()

    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    ax1.plot(tt, XX)
    ax1.set_ylabel("x(t)")

    ax2.plot(tt, YY)
    ax2.set_ylabel("y(t)")

    ax3.plot(tt, PHI)
    ax3.set_ylabel(r"\phi(t)")

    plt.show()
    exit()


def init():
    # add rectangle
    ax.add_patch(rect1)

    ax.axis('equal')
    # ax.set_xlim(-3, 3)
    ax.set_xlim(-2, 2)
    ax.grid(1)
    ln.set_data(XX[:0], YY[:0])

    return ln, rect1


def update(frame_index):
    """
    This function updates the frames
    """

    i = int(frame_index)
    print(i)

    # rotate the rectangle and translate it to the desired point on the path
    rot = mpl.transforms.Affine2D().rotate(PHI[i]).translate(XX[i], YY[i])

    # transformation from data coords to displaycoords
    rot += ax.transData

    # apply transorm
    rect1.set_transform(rot)

    # draw the path
    ln.set_data(XX[:i], YY[:i])

    # save the frame if the appropriate flag is set
    if save_images:
        plt.savefig("{:04d}.png".format(i))

    return ln, rect1


ff = np.arange(N)
ani = FuncAnimation(fig, update, frames=ff,
                    init_func=init, blit=True, interval=.1, repeat=False)


plt.show()
