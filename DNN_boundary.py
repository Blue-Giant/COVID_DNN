import tensorflow as tf
import numpy as np

# remark: The formal parameters of uleft, uright, utop and ubottom should keep the shape [None, dim]
# where dim represents the dimension of problem. For example dim=2
# U_left = tf.reshape(u_left(XY_left_bd[:, 0], XY_left_bd[:, 1]), shape=[-1, 1])
# tf.square(UNN_left - U_left)


# -------------------------------------------------------------------------------------------------------
#                                                  0阶导数边界
# -------------------------------------------------------------------------------------------------------
def deal_0derivatives2NN_1d(UNN_left, UNN_right, uleft, uright):
    loss = tf.square(UNN_left-uleft) + tf.square(UNN_right-uright)
    return loss


def deal_0derivatives2NN_2d(UNN_left, UNN_right, UNN_top, UNN_bottom, uleft, uright, utop, ubottom):
    loss = tf.square(UNN_left - uleft) + tf.square(UNN_right - uright) + tf.square(UNN_top - utop) + \
           tf.square(UNN_bottom - ubottom)
    return loss


def deal_0derivatives2NN_3d(UNN_left, UNN_right, UNN_front, UNN_behiind, UNN_top, UNN_bottom, uleft, uright, ufront,
                            ubehind, utop, ubottom):
    loss = tf.square(UNN_left - uleft) + tf.square(UNN_right - uright) + \
           tf.square(UNN_front - ufront) + tf.square(UNN_behiind - ubehind) + \
           tf.square(UNN_top - utop) + tf.square(UNN_bottom - ubottom)
    return loss


# -------------------------------------------------------------------------------------------------------
#                                                  1阶导数边界
# -------------------------------------------------------------------------------------------------------
def deal_1derivatives2NN_1d(UNN_left, UNN_right, uleft, uright, xleft, xright):
    dULeft2NN = tf.gradients(UNN_left, xleft)[0]
    dURight2NN = tf.gradients(UNN_right, xright)[0]
    loss = tf.square(dULeft2NN-uleft) + tf.square(dURight2NN-uright)
    return loss


def deal_1derivative2NN_2d(UNN_bd, ubd_1derivative, varible_xy, nameBoundary=None):
    if str.lower(nameBoundary) == 'left':
        dUNN_bd = tf.gradients(UNN_bd, varible_xy)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [0], axis=-1)
    elif str.lower(nameBoundary) == 'right':
        dUNN_bd = tf.gradients(UNN_bd, varible_xy)[0]
        dUNNbd = tf.gather(dUNN_bd, [0], axis=-1)
    elif str.lower(nameBoundary) == 'bottom':
        dUNN_bd = tf.gradients(UNN_bd, varible_xy)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [1], axis=-1)
    elif str.lower(nameBoundary) == 'top':
        dUNN_bd = tf.gradients(UNN_bd, varible_xy)[0]
        dUNNbd = tf.gather(dUNN_bd, [1], axis=-1)
    loss = tf.square(dUNNbd - ubd_1derivative)
    return loss


def deal_1derivatives2NN_2d(UNN_left, UNN_right, UNN_top, UNN_bottom, uleft_1deriva, uright_1deriva,
                            utop_1deriva, ubottom_1deriva, xyleft, xyright, xytop, xybottom):
    dULeft2NN = tf.gradients(UNN_left, xyleft)[0]
    dULeft_NN = -1.0*tf.gather(dULeft2NN, [0], axis=-1)

    dURight2NN = tf.gradients(UNN_right, xyright)[0]
    dURight_NN = tf.gather(dURight2NN, [0], axis=-1)

    dUTop2NN = tf.gradients(UNN_top, xytop)[0]
    ddUTop_NN = tf.gather(dUTop2NN, [1], axis=-1)

    dUBottom2NN = tf.gradients(UNN_bottom, xybottom)[0]
    dUBottom_NN = -1.0 * tf.gather(dUBottom2NN, [1], axis=-1)

    loss = tf.square(dULeft_NN - uleft_1deriva) + tf.square(dURight_NN - uright_1deriva) + \
                        tf.square(ddUTop_NN - utop_1deriva) + tf.square(dUBottom_NN - ubottom_1deriva)
    return loss


def deal_1derivative2NN_3d(UNN_bd, ubd_1derivative, varible_xyz, nameBoundary=None):
    if str.lower(nameBoundary) == 'left':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [1], axis=-1)
    elif str.lower(nameBoundary) == 'right':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = tf.gather(dUNN_bd, [1], axis=-1)
    elif str.lower(nameBoundary) == 'bottom':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [2], axis=-1)
    elif str.lower(nameBoundary) == 'top':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = tf.gather(dUNN_bd, [2], axis=-1)
    elif str.lower(nameBoundary) == 'front':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = tf.gather(dUNN_bd, [0], axis=-1)
    elif str.lower(nameBoundary) == 'behind':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [0], axis=-1)
    loss = tf.square(dUNNbd - ubd_1derivative)
    return loss


def deal_1derivatives2NN_3d(UNN_left, UNN_right, UNN_front, UNN_behind, UNN_top, UNN_bottom, uleft_1deriva,
                            uright_1deriva, ufront_1deriva, ubehind_1deriva, utop_1deriva, ubottom_1deriva,
                            xyzleft, xyzright, xyzfront, xyzbehind, xyztop, xyzbottom):
    dULeft2NN = tf.gradients(UNN_left, xyzleft)[0]
    dULeft_NN = -1.0 * tf.gather(dULeft2NN, [1], axis=-1)

    dURight2NN = tf.gradients(UNN_right, xyzright)[0]
    dURight_NN = tf.gather(dURight2NN, [1], axis=-1)

    dUFront2NN = tf.gradients(UNN_front, xyzfront)[0]
    dUFront_NN = -1.0 * tf.gather(dUFront2NN, [1], axis=-1)

    dUBehind2NN = tf.gradients(UNN_behind, xyzbehind)[0]
    dUBehind_NN = tf.gather(dUBehind2NN, [1], axis=-1)

    dUTop2NN = tf.gradients(UNN_top, xyztop)[0]
    dUTop_NN = -1.0 * tf.gather(dUTop2NN, [1], axis=-1)

    dUBottom2NN = tf.gradients(UNN_bottom, xyzbottom)[0]
    dUBottom_NN = tf.gather(dUBottom2NN, [1], axis=-1)

    loss = tf.square(dULeft_NN - uleft_1deriva) + tf.square(dURight_NN - uright_1deriva) + \
           tf.square(dUFront_NN - ufront_1deriva) + tf.square(dUBehind_NN - ubehind_1deriva) + \
           tf.square(dUTop_NN - utop_1deriva) + tf.square(dUBottom_NN - ubottom_1deriva)
    return loss


# -------------------------------------------------------------------------------------------------------
#                                                  2阶导数边界
# -------------------------------------------------------------------------------------------------------
def deal_2derivatives2NN_1d(UNN_left, UNN_right, uleft_2deriva, uright_2deriva, xleft, xright):
    dULeft2NN = tf.gradients(UNN_left, xleft)[0]
    dULeft_NN = tf.gradients(dULeft2NN, xleft)[0]
    dURight2NN = tf.gradients(UNN_right, xright)[0]
    dURight_NN = tf.gradients(dURight2NN, xright)[0]
    loss = tf.square(dULeft_NN-uleft_2deriva) + tf.square(dURight_NN-uright_2deriva)
    return loss


def deal_2derivative2NN_2d(UNN_bd, ubd_2derivative, varible_xy, nameBoundary=None):
    if str.lower(nameBoundary) == 'left':
        dUNN_bd = tf.gradients(UNN_bd, varible_xy)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [0], axis=-1)
    elif str.lower(nameBoundary) == 'right':
        dUNN_bd = tf.gradients(UNN_bd, varible_xy)[0]
        dUNNbd = tf.gather(dUNN_bd, [0], axis=-1)
    elif str.lower(nameBoundary) == 'bottom':
        dUNN_bd = tf.gradients(UNN_bd, varible_xy)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [1], axis=-1)
    elif str.lower(nameBoundary) == 'top':
        dUNN_bd = tf.gradients(UNN_bd, varible_xy)[0]
        dUNNbd = tf.gather(dUNN_bd, [1], axis=-1)
    loss = tf.square(dUNNbd - ubd_2derivative)
    return loss


def deal_2derivatives2NN_2d(UNN_left, UNN_right, UNN_top, UNN_bottom, uleft_2deriva, uright_2deriva,
                            utop_2deriva, ubottom_2deriva, xyleft, xyright, xytop, xybottom):
    dULeft2NN = tf.gradients(UNN_left, xyleft)[0]
    dULeft_NN_x = tf.gather(dULeft2NN, [0], axis=-1)
    dULeft_NN_y = tf.gather(dULeft2NN, [1], axis=-1)
    ddUL_ReLU_x = tf.gradients(dULeft_NN_x, xyleft)[0]
    ULeft_NN_xx = tf.gather(ddUL_ReLU_x, [0], axis=-1)
    ddUL_ReLU_y = tf.gradients(dULeft_NN_y, xyleft)[0]
    ULeft_NN_yy = tf.gather(ddUL_ReLU_y, [1], axis=-1)
    laplaceULeft_NN = ULeft_NN_xx + ULeft_NN_yy

    dURight2NN = tf.gradients(UNN_right, xyright)[0]
    dURight_NN_x = tf.gather(dURight2NN, [0], axis=-1)
    dURight_NN_y = tf.gather(dURight2NN, [1], axis=-1)
    ddUR_ReLU_x = tf.gradients(dURight_NN_x, xyright)[0]
    URight_NN_xx = tf.gather(ddUR_ReLU_x, [0], axis=-1)
    ddUR_ReLU_y = tf.gradients(dURight_NN_y, xyright)[0]
    URight_NN_yy = tf.gather(ddUR_ReLU_y, [1], axis=-1)
    laplaceURight_NN = URight_NN_xx + URight_NN_yy

    dUBottom_NN = tf.gradients(UNN_bottom, xybottom)[0]
    dUBottom_NN_x = tf.gather(dUBottom_NN, [0], axis=-1)
    dUBottom_NN_y = tf.gather(dUBottom_NN, [1], axis=-1)
    ddUB_ReLU_x = tf.gradients(dUBottom_NN_x, xybottom)[0]
    UBottom_NN_xx = tf.gather(ddUB_ReLU_x, [0], axis=-1)
    ddUB_ReLU_y = tf.gradients(dUBottom_NN_y, xybottom)[0]
    UBottom_NN_yy = tf.gather(ddUB_ReLU_y, [1], axis=-1)
    laplaceUBottom_NN = UBottom_NN_xx + UBottom_NN_yy

    dUTop_NN = tf.gradients(UNN_top, xytop)[0]
    dUTop_NN_x = tf.gather(dUTop_NN, [0], axis=-1)
    dUTop_NN_y = tf.gather(dUTop_NN, [1], axis=-1)
    ddUT_ReLU_x = tf.gradients(dUTop_NN_x, xytop)[0]
    UTop_NN_xx = tf.gather(ddUT_ReLU_x, [0], axis=-1)
    ddUT_ReLU_y = tf.gradients(dUTop_NN_y, xytop)[0]
    UTop_NN_yy = tf.gather(ddUT_ReLU_y, [1], axis=-1)
    laplaceUTop_NN = UTop_NN_xx + UTop_NN_yy

    loss = tf.square(laplaceULeft_NN - uleft_2deriva) + tf.square(laplaceURight_NN - uright_2deriva) + \
                        tf.square(laplaceUTop_NN - utop_2deriva) + tf.square(laplaceUBottom_NN - ubottom_2deriva)
    return loss


def deal_2derivative2NN_3d(UNN_bd, ubd_1derivative, varible_xyz, nameBoundary=None):
    if str.lower(nameBoundary) == 'left':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [1], axis=-1)
    elif str.lower(nameBoundary) == 'right':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = tf.gather(dUNN_bd, [1], axis=-1)
    elif str.lower(nameBoundary) == 'bottom':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [2], axis=-1)
    elif str.lower(nameBoundary) == 'top':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = tf.gather(dUNN_bd, [2], axis=-1)
    elif str.lower(nameBoundary) == 'front':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = tf.gather(dUNN_bd, [0], axis=-1)
    elif str.lower(nameBoundary) == 'behind':
        dUNN_bd = tf.gradients(UNN_bd, varible_xyz)[0]
        dUNNbd = -1.0*tf.gather(dUNN_bd, [0], axis=-1)
    loss = tf.square(dUNNbd - ubd_1derivative)
    return loss


def deal_2derivatives2NN_3d(UNN_left, UNN_right, UNN_front, UNN_behind, UNN_top, UNN_bottom, uleft_1deriva,
                            uright_1deriva, ufront_1deriva, ubehind_1deriva, utop_1deriva, ubottom_1deriva,
                            xyzleft, xyzright, xyzfront, xyzbehind, xyztop, xyzbottom):
    dULeft2NN = tf.gradients(UNN_left, xyzleft)[0]
    dULeft_NN_x = tf.gather(dULeft2NN, [0], axis=-1)
    dULeft_NN_y = tf.gather(dULeft2NN, [1], axis=-1)
    dULeft_NN_z = tf.gather(dULeft2NN, [2], axis=-1)
    ddUL_NN_x = tf.gradients(dULeft_NN_x, xyzleft)[0]
    ULeft_NN_xx = tf.gather(ddUL_NN_x, [0], axis=-1)
    ddUL_NN_y = tf.gradients(dULeft_NN_y, xyzleft)[0]
    ULeft_NN_yy = tf.gather(ddUL_NN_y, [1], axis=-1)
    ddUL_NN_z = tf.gradients(dULeft_NN_z, xyzleft)[0]
    ULeft_NN_zz = tf.gather(ddUL_NN_z, [2], axis=-1)
    laplaceULeft_NN = ULeft_NN_xx + ULeft_NN_yy + ULeft_NN_zz

    dURight2NN = tf.gradients(UNN_right, xyzright)[0]
    dURight_NN_x = tf.gather(dURight2NN, [0], axis=-1)
    dURight_NN_y = tf.gather(dURight2NN, [1], axis=-1)
    dURight_NN_z = tf.gather(dURight2NN, [2], axis=-1)
    ddUR_NN_x = tf.gradients(dURight_NN_x, xyzright)[0]
    URight_NN_xx = tf.gather(ddUR_NN_x, [0], axis=-1)
    ddUR_NN_y = tf.gradients(dURight_NN_y, xyzright)[0]
    URight_NN_yy = tf.gather(ddUR_NN_y, [1], axis=-1)
    ddUR_NN_z = tf.gradients(dURight_NN_z, xyzright)[0]
    URight_NN_zz = tf.gather(ddUR_NN_z, [2], axis=-1)
    laplaceURight_NN = URight_NN_xx + URight_NN_yy + URight_NN_zz

    dUFront2NN = tf.gradients(UNN_front, xyzfront)[0]
    dUFront_NN_x = tf.gather(dUFront2NN, [0], axis=-1)
    dUFront_NN_y = tf.gather(dUFront2NN, [1], axis=-1)
    dUFront_NN_z = tf.gather(dUFront2NN, [2], axis=-1)
    ddUF_NN_x = tf.gradients(dUFront_NN_x, xyzfront)[0]
    UFront_NN_xx = tf.gather(ddUF_NN_x, [0], axis=-1)
    ddUF_NN_y = tf.gradients(dUFront_NN_y, xyzfront)[0]
    UFront_NN_yy = tf.gather(ddUF_NN_y, [1], axis=-1)
    ddUF_NN_z = tf.gradients(dUFront_NN_z, xyzfront)[0]
    UFront_NN_zz = tf.gather(ddUF_NN_z, [2], axis=-1)
    laplaceUFront_NN = UFront_NN_xx + UFront_NN_yy + UFront_NN_zz

    dUBehind2NN = tf.gradients(UNN_behind, xyzbehind)[0]
    dUBehind_NN_x = tf.gather(dUBehind2NN, [0], axis=-1)
    dUBehind_NN_y = tf.gather(dUBehind2NN, [1], axis=-1)
    dUBehind_NN_z = tf.gather(dUBehind2NN, [2], axis=-1)
    ddUBE_NN_x = tf.gradients(dUBehind_NN_x, xyzbehind)[0]
    UBehind_NN_xx = tf.gather(ddUBE_NN_x, [0], axis=-1)
    ddUBE_NN_y = tf.gradients(dUBehind_NN_y, xyzbehind)[0]
    UBehind_NN_yy = tf.gather(ddUBE_NN_y, [1], axis=-1)
    ddUBE_NN_z = tf.gradients(dUBehind_NN_z, xyzbehind)[0]
    UBehind_NN_zz = tf.gather(ddUBE_NN_z, [2], axis=-1)
    laplaceUBehind_NN = UBehind_NN_xx + UBehind_NN_yy + UBehind_NN_zz

    dUTop_NN = tf.gradients(UNN_top, xyztop)[0]
    dUTop_NN_x = tf.gather(dUTop_NN, [0], axis=-1)
    dUTop_NN_y = tf.gather(dUTop_NN, [1], axis=-1)
    dUTop_NN_z = tf.gather(dUTop_NN, [2], axis=-1)
    ddUT_NN_x = tf.gradients(dUTop_NN_x, xyztop)[0]
    UTop_NN_xx = tf.gather(ddUT_NN_x, [0], axis=-1)
    ddUT_NN_y = tf.gradients(dUTop_NN_y, xyztop)[0]
    UTop_NN_yy = tf.gather(ddUT_NN_y, [1], axis=-1)
    ddUT_NN_z = tf.gradients(dUTop_NN_z, xyztop)[0]
    UTop_NN_zz = tf.gather(ddUT_NN_z, [2], axis=-1)
    laplaceUTop_NN = UTop_NN_xx + UTop_NN_yy + UTop_NN_zz

    dUBottom_NN = tf.gradients(UNN_bottom, xyzbottom)[0]
    dUBottom_NN_x = tf.gather(dUBottom_NN, [0], axis=-1)
    dUBottom_NN_y = tf.gather(dUBottom_NN, [1], axis=-1)
    dUBottom_NN_z = tf.gather(dUBottom_NN, [2], axis=-1)
    ddUB_NN_x = tf.gradients(dUBottom_NN_x, xyzbottom)[0]
    UBottom_NN_xx = tf.gather(ddUB_NN_x, [0], axis=-1)
    ddUB_NN_y = tf.gradients(dUBottom_NN_y, xyzbottom)[0]
    UBottom_NN_yy = tf.gather(ddUB_NN_y, [1], axis=-1)
    ddUB_NN_z = tf.gradients(dUBottom_NN_z, xyzbottom)[0]
    UBottom_NN_zz = tf.gather(ddUB_NN_z, [2], axis=-1)
    laplaceUBottom_NN = UBottom_NN_xx + UBottom_NN_yy + UBottom_NN_zz

    loss = tf.square(laplaceULeft_NN - uleft_1deriva) + tf.square(laplaceURight_NN - uright_1deriva) + \
           tf.square(laplaceUFront_NN - ufront_1deriva) + tf.square(laplaceUBehind_NN - ubehind_1deriva) + \
           tf.square(laplaceUTop_NN - utop_1deriva) + tf.square(laplaceUBottom_NN - ubottom_1deriva)
    return loss