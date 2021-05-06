import DNN_base
import tensorflow as tf


def init_DNN(size2in=1, size2out=1, hiddens=None, scope='flag', opt2Init='General_Init'):
    if 'GENERAL_INIT' == str.upper(opt2Init):
        Ws, Bs = DNN_base.Generally_Init_NN(size2in, size2out, hiddens, Flag=scope)
    elif 'Truncated_normal_init' == str.title(opt2Init):
        Ws, Bs = DNN_base.Truncated_normal_init_NN(size2in, size2out, hiddens, Flag=scope)
    elif 'XAVIER_INIT' == str.upper(opt2Init):
        Ws, Bs = DNN_base.Xavier_init_NN(size2in, size2out, hiddens, Flag=scope)
    elif 'XAVIER_INIT_FOURIER' == str.upper(opt2Init):
        Ws, Bs = DNN_base.Xavier_init_NN_Fourier(size2in, size2out, hiddens, Flag=scope)

    return Ws, Bs


def PDE_DNN(x=None, hiddenLayer=None, Weigths=None, Biases=None, DNNmodel=None, activation=None, freqs=[1]):
    if 'DNN' == str.upper(DNNmodel):
        UNN = DNN_base.DNN(x, Weigths, Biases, hiddenLayer, activate_name=activation)
    elif 'DNN_SCALE' == str.upper(DNNmodel):
        UNN = DNN_base.DNN_scale(x, Weigths, Biases, hiddenLayer, freq_frag=freqs, activate_name=activation)
    elif 'DNN_ADAPT_SCALE' == str.upper(DNNmodel):
        UNN = DNN_base.DNN_adapt_scale(x, Weigths, Biases, hiddenLayer, freq_frag=freqs, activate_name=activation)
    elif 'DNN_SINORCOS_BASE' == str.upper(DNNmodel):
        UNN = DNN_base.DNN_Sine0rCos_Base(x, Weigths, Biases, hiddenLayer, freq_frag=freqs, activate_name=activation)
    elif 'DNN_FOURIERBASE' == str.upper(DNNmodel):
        UNN = DNN_base.DNN_FourierBase(x, Weigths, Biases, hiddenLayer, freq_frag=freqs, activate_name=activation)
    return UNN


# 求 U 关于变量 x 哪个维度的高阶导数
def diff_UNN2x(UNN=None, x=None, order2deri=1, axis=0, dim2x=1):
    if 1 == order2deri:
        dUNN = tf.gradients(UNN, x)[0]
        return dUNN
    elif 2 == order2deri:
        dUNN = tf.gradients(UNN, x)[0]
        if 1 == dim2x:
            ddUNN = tf.gradients(dUNN, x)[0]
        elif 2 == dim2x:
            if 0 == axis:
                dUNNx = tf.gather(dUNN, [0], axis=-1)
                ddUNN1 = tf.gradients(dUNNx, x)[0]
                ddUNN = tf.gather(ddUNN1, [0], axis=-1)
            if 1 == axis:
                dUNNy = tf.gather(dUNN, [1], axis=-1)
                ddUNN2 = tf.gradients(dUNNy, x)[0]
                ddUNN = tf.gather(ddUNN2, [1], axis=-1)
        elif 3 == dim2x:
            if 0 == axis:
                dUNNx = tf.gather(dUNN, [0], axis=-1)
                ddUNN1 = tf.gradients(dUNNx, x)[0]
                ddUNN = tf.gather(ddUNN1, [0], axis=-1)
            if 1 == axis:
                dUNNy = tf.gather(dUNN, [1], axis=-1)
                ddUNN2 = tf.gradients(dUNNy, x)[0]
                ddUNN = tf.gather(ddUNN2, [1], axis=-1)
            if 2 == axis:
                dUNNz = tf.gather(dUNN, [2], axis=-1)
                ddUNN3 = tf.gradients(dUNNz, x)[0]
                ddUNN = tf.gather(ddUNN3, [2], axis=-1)
        return ddUNN


# 求 U 关于变量 x 哪个维度的高阶导数
def unstack_diff_UNN(U=None, x=None, order2deri=1, dim_info=0, dim2x=1):
    if 1 == order2deri:
        dUNN = tf.gradients(U, x)[0]
        return dUNN
    elif 2 == order2deri:
        dUNN = tf.gradients(U, x)[0]
        if 1 == dim2x:
            ddUNN = tf.gradients(dUNN, x)[0]
        else:
            dUNN_dim = tf.gather(dUNN, [dim_info], axis=-1)
            ddUNN = tf.gradients(dUNN_dim, x[dim_info])[0]
        return ddUNN
    elif 3 == order2deri:
        dUNN = tf.gradients(U, x)[0]
        if 1 == dim2x:
            ddUNN = tf.gradients(dUNN, x)[0]
            dddUNN = tf.gradients(ddUNN, x)[0]
        else:
            dUNN_dim = tf.gather(dUNN, [dim_info], axis=-1)
            ddUNN = tf.gradients(dUNN_dim, x[dim_info])[0]
            dddUNN = tf.gradients(ddUNN, x[dim_info])[0]
        return dddUNN
    elif 4 == order2deri:
        dUNN = tf.gradients(U, x)[0]
        if 1 == dim2x:
            ddUNN = tf.gradients(dUNN, x)[0]
            dddUNN = tf.gradients(ddUNN, x)[0]
            ddddUNN = tf.gradients(dddUNN, x)[0]
        else:
            dUNN_dim = tf.gather(dUNN, [dim_info], axis=-1)
            ddUNN = tf.gradients(dUNN_dim, x[dim_info])[0]
            dddUNN = tf.gradients(ddUNN, x[dim_info])[0]
            ddddUNN = tf.gradients(dddUNN, x[dim_info])[0]
        return ddddUNN


# 求 U 关于变量 x 混合导数, axis='01'指对xy，axis='02'指对xz, axis='12'指对yz.
# 由于混合导数至少为二阶导数，order2deri 应 >=2
def diff_UNN2xy(UNN=None, order2deri=2, x=None, axis='01', dim2x=1):
    if 1 == dim2x:
        return
    if 1 == order2deri:
        return
    assert(order2deri >= 2)
    if 2 == order2deri:
        dUNN = tf.gradients(UNN, x)[0]
        if 2 == dim2x:
            assert (axis == '01')
            dUNNxy = tf.gather(dUNN, [0], axis=-1)
            ddUNN1 = tf.gradients(dUNNxy, x)[0]
            ddUNN = tf.gather(ddUNN1, [1], axis=-1)
        if 3 == dim2x:
            if axis == '01':
                dUNNxy = tf.gather(dUNN, [0], axis=-1)
                ddUNN1 = tf.gradients(dUNNxy, x)[0]
                ddUNN = tf.gather(ddUNN1, [1], axis=-1)
            if axis == '02':
                dUNNxz = tf.gather(dUNN, [0], axis=-1)
                ddUNN1 = tf.gradients(dUNNxz, x)[0]
                ddUNN = tf.gather(ddUNN1, [2], axis=-1)
            if axis == '12':
                dUNNyz = tf.gather(dUNN, [1], axis=-1)
                ddUNN1 = tf.gradients(dUNNyz, x)[0]
                ddUNN = tf.gather(ddUNN1, [2], axis=-1)
        return ddUNN