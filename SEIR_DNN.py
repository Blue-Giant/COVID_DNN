"""
@author: LXA
Benchmark Code of Biharmonic equations.

"""
import os
import sys
import tensorflow as tf
import numpy as np
import time
import platform
import shutil
import DNN_base
import DNN_tools
import DNN_data
import plotData
import saveData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation type for problem: %s\n' % (R_dic['eqs_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('activate function: %s\n' % str(R_dic['act_name']), log_fileout)
    DNN_tools.log_string('hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)

    if 1 == R['Dirichlet_boundary']:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Dirichlet boundary'), log_fileout)
    else:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Navier boundary'), log_fileout)
    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_bd_penalty']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)

    if R_dic['variational_loss'] == 1:
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: original function loss\n', log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, temp_penalty_bd, penalty_wb, loss_it_tmp, loss_bd_tmp, loss_bd2_tmp,
                        loss_tmp, train_mse_tmp, train_res_tmp, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('boundary penalty: %f' % temp_penalty_bd)
    print('weights and biases with  penalty: %f' % penalty_wb)
    print('loss_it for training: %.10f' % loss_it_tmp)
    print('loss_bd for training: %.10f' % loss_bd_tmp)
    print('loss_bd to derivative for training: %.10f' % loss_bd2_tmp)
    print('total loss for training: %.10f' % loss_tmp)
    print('function mean square error for training: %.10f' % train_mse_tmp)
    print('function residual error for training: %.10f\n' % train_res_tmp)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('boundary penalty: %f' % temp_penalty_bd, log_out)
    DNN_tools.log_string('weights and biases with  penalty: %f' % penalty_wb, log_out)
    DNN_tools.log_string('loss_it for training: %.10f' % loss_it_tmp, log_out)
    DNN_tools.log_string('loss_bd for training: %.10f' % loss_bd_tmp, log_out)
    DNN_tools.log_string('loss_bd to derivative for training: %.10f' % loss_bd2_tmp, log_out)
    DNN_tools.log_string('total loss for training: %.10f' % loss_tmp, log_out)
    DNN_tools.log_string('function mean square error for training: %.10f' % train_mse_tmp, log_out)
    DNN_tools.log_string('function residual error for training: %.10f\n' % train_res_tmp, log_out)


def print_and_log2test(mse2test, res2test, log_out=None):
    print('mean square error of predict and real for testing: %.10f' % mse2test)
    print('residual error of predict and real for testing: %.10f\n' % res2test)

    DNN_tools.log_string('mean square error of predict and real for testing: %.10f' % mse2test, log_out)
    DNN_tools.log_string('residual error of predict and real for testing: %.10f\n\n' % res2test, log_out)


def solve_COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    batchsize_it = R['batch_size2interior']
    # batchsize_bd = R['batch_size2boundary']
    # bd_penalty_init = R['init_bd_penalty']            # Regularization parameter for boundary conditions
    wb_penalty = R['regular_weight']                  # Regularization parameter for weights
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']
    act_func = R['act_name']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # 问题区域，每个方向设置为一样的长度
    region_lb = 0.0
    region_rt = 1.0
    if str.lower(R['eqs_type']) == 'general_biharmonic':
        mu = 1.0
        # laplace laplace u = f
        f, u_true, u_left, u_right, u_bottom, u_top = Biharmonic_eqs.get_biharmonic_infos_2D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, laplace_name=R['eqs_name'])

    flag2S = 'WB2S'
    flag2E = 'WB2S'
    flag2I = 'WB2S'
    flag2R = 'WB2S'
    hidden_layers = R['hidden_layers']
    Weight2S, Bias2S = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2S)
    Weight2E, Bias2E = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2E)
    Weight2I, Bias2I = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2I)
    Weight2R, Bias2R = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2R)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            T_it = tf.placeholder(tf.float32, name='XY_it', shape=[None, 1])
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')
            if 'PDE_DNN' == str.upper(R['model']):
                S_NN = DNN_base.PDE_DNN(T_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func)
                E_NN = DNN_base.PDE_DNN(T_it, Weight2E, Bias2E, hidden_layers, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN(T_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN(T_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func)
            elif 'PDE_DNN_BN' == str.upper(R['model']):
                S_NN = DNN_base.PDE_DNN_BN(T_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func, is_training=train_opt)
                E_NN = DNN_base.PDE_DNN_BN(T_it, Weight2E, Bias2E, hidden_layers, activate_name=act_func, is_training=train_opt)
                I_NN = DNN_base.PDE_DNN_BN(T_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func, is_training=train_opt)
                R_NN = DNN_base.PDE_DNN_BN(T_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func, is_training=train_opt)
            elif 'PDE_DNN_SCALE' == str.upper(R['model']):
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                S_NN = DNN_base.PDE_DNN_scale(T_it, Weight2S, Bias2S, hidden_layers, freq, activate_name=act_func)
                E_NN = DNN_base.PDE_DNN_scale(T_it, Weight2E, Bias2E, hidden_layers, freq, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN_scale(T_it, Weight2I, Bias2I, hidden_layers, freq, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN_scale(T_it, Weight2R, Bias2R, hidden_layers, freq, activate_name=act_func)

            N_NN = S_NN + E_NN + I_NN + R_NN

            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dE_NN2t = tf.gradients(E_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]

            temp_snn2t = -beta*S_NN*I_NN + folw_in -mu*S_NN
            temp_enn2t = beta*S_NN*I_NN - (mu*+k)* E_NN
            temp_inn2t = k*E_NN -(gamma+mu)*I_NN
            temp_rnn2t = gamma*I_NN - mu*R_NN

            loss_temp = tf.square(dS_NN2t-temp_snn2t) + tf.square(dE_NN2t-temp_enn2t) + tf.square(dI_NN2t-temp_inn2t) + \
                        tf.square(dR_NN2t-temp_rnn2t)
            loss_dt2NNs = tf.reduce_mean(loss_temp)

            Loss2S = 1