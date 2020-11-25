"""
@author: LXA
Benchmark Code of SEIRD model
2020-11-13
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
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('activate function: %s\n' % str(R_dic['act_name']), log_fileout)
    DNN_tools.log_string('hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    DNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']),
                             log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']),
                             log_fileout)

    DNN_tools.log_string(
        'Initial penalty for difference of predict and true: %s\n' % str(R_dic['init_penalty2predict_true']),
        log_fileout)

    DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, temp_penalty_nt, penalty_wb2s, penalty_wb2e, penalty_wb2i, penalty_wb2r,
                        loss_s, loss_e, loss_i, loss_r, loss_n, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('penalty for difference of predict and true : %f' % temp_penalty_nt)
    print('penalty weights and biases for S: %f' % penalty_wb2s)
    print('penalty weights and biases for S: %f' % penalty_wb2e)
    print('penalty weights and biases for I: %f' % penalty_wb2i)
    print('penalty weights and biases for R: %f' % penalty_wb2r)
    print('loss for S: %.10f' % loss_s)
    print('loss for S: %.10f' % loss_e)
    print('loss for I: %.10f' % loss_i)
    print('loss for R: %.10f' % loss_r)
    print('total loss: %.10f' % loss_n)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('penalty for difference of predict and true : %f' % temp_penalty_nt, log_out)
    DNN_tools.log_string('penalty weights and biases for S: %f' % penalty_wb2s, log_out)
    DNN_tools.log_string('penalty weights and biases for S: %f' % penalty_wb2e, log_out)
    DNN_tools.log_string('penalty weights and biases for I: %f' % penalty_wb2i, log_out)
    DNN_tools.log_string('penalty weights and biases for R: %f' % penalty_wb2r, log_out)
    DNN_tools.log_string('loss for S: %.10f' % loss_s, log_out)
    DNN_tools.log_string('loss for S: %.10f' % loss_e, log_out)
    DNN_tools.log_string('loss for I: %.10f' % loss_i, log_out)
    DNN_tools.log_string('loss for R: %.10f' % loss_r, log_out)
    DNN_tools.log_string('total loss: %.10f' % loss_n, log_out)


def solve_SEIRD2COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    batchsize_it = R['batch_size2interior']
    # batchsize_bd = R['batch_size2boundary']
    bd_penalty_init = R['init_bd_penalty']            # Regularization parameter for boundary conditions
    wb_penalty = R['regular_weight']                  # Regularization parameter for weights
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']
    act_func = R['act_name']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # parameters for Convid
    alpha = 1.0
    beta_i = 1.0
    beta_e = 1.0
    A = 1.0
    mu = 1.0
    sigma = 1.0
    phi_e = 1.0
    phi_r = 1.0
    phi_d = 1.0
    nu_s = 1.0
    nu_e = 1.0
    nu_i = 1.0
    nu_r = 1.0

    # 问题区域，每个方向设置为一样的长度
    region_lb = 0.0
    region_rt = 1.0
    if str.lower(R['eqs_type']) == 'convid':
        S_true, E_true, I_true, R_true, D_true, N_true = Biharmonic_eqs.get_biharmonic_infos_2D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt,
            laplace_name=R['eqs_name'])

    flag2S = 'WB2S'
    flag2E = 'WB2E'
    flag2I = 'WB2I'
    flag2R = 'WB2R'
    flag2D = 'WB2D'
    hidden_layers = R['hidden_layers']
    Weight2S, Bias2S = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2S)
    Weight2E, Bias2E = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2E)
    Weight2I, Bias2I = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2I)
    Weight2R, Bias2R = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2R)
    Weight2D, Bias2D = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2D)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            T_it = tf.placeholder(tf.float32, name='XY_it', shape=[None, 1])
            XY_it = tf.placeholder(tf.float32, name='XY_it', shape=[None, 2])
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')
            TXY_it = tf.concat([T_it, XY_it], axis=-1)
            if 'PDE_DNN' == str.upper(R['model']):
                S_NN = DNN_base.PDE_DNN(TXY_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func)
                E_NN = DNN_base.PDE_DNN(TXY_it, Weight2E, Bias2E, hidden_layers, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN(TXY_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN(TXY_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func)
                D_NN = DNN_base.PDE_DNN(TXY_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func)
            elif 'PDE_DNN_BN' == str.upper(R['model']):
                S_NN = DNN_base.PDE_DNN_BN(TXY_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func, is_training=train_opt)
                E_NN = DNN_base.PDE_DNN_BN(TXY_it, Weight2E, Bias2E, hidden_layers, activate_name=act_func, is_training=train_opt)
                I_NN = DNN_base.PDE_DNN_BN(TXY_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func, is_training=train_opt)
                R_NN = DNN_base.PDE_DNN_BN(TXY_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func, is_training=train_opt)
                D_NN = DNN_base.PDE_DNN_BN(TXY_it, Weight2D, Bias2D, hidden_layers, activate_name=act_func,
                                           is_training=train_opt)
            elif 'PDE_DNN_SCALE' == str.upper(R['model']):
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                S_NN = DNN_base.PDE_DNN_scale(TXY_it, Weight2S, Bias2S, hidden_layers, freq, activate_name=act_func)
                E_NN = DNN_base.PDE_DNN_scale(TXY_it, Weight2E, Bias2E, hidden_layers, freq, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN_scale(TXY_it, Weight2I, Bias2I, hidden_layers, freq, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN_scale(TXY_it, Weight2R, Bias2R, hidden_layers, freq, activate_name=act_func)
                D_NN = DNN_base.PDE_DNN_scale(TXY_it, Weight2D, Bias2D, hidden_layers, freq, activate_name=act_func)

            X_it = tf.reshape(XY_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XY_it[:, 1], shape=[-1, 1])

            N_NN = S_NN + E_NN + I_NN + R_NN

            Loss2SNN = tf.reduce_mean(tf.square(S_NN - S_true))
            Loss2ENN = tf.reduce_mean(tf.square(E_NN - E_true))
            Loss2INN = tf.reduce_mean(tf.square(I_NN - I_true))
            Loss2RNN = tf.reduce_mean(tf.square(R_NN - R_true))
            Loss2DNN = tf.reduce_mean(tf.square(D_NN - D_true))

            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dE_NN2t = tf.gradients(E_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]
            dD_NN2t = tf.gradients(D_NN, T_it)[0]

            gradN_NN2xy = tf.gradients(N_NN, XY_it)[0]
            dN_NN2x = tf.gather(gradN_NN2xy, [0], axis=-1)
            dN_NN2y = tf.gather(gradN_NN2xy, [1], axis=-1)

            gradS_NN2xy = tf.gradients(S_NN, XY_it)[0]
            dS_NN2x = tf.gather(gradS_NN2xy, [0], axis=-1)
            dS_NN2y = tf.gather(gradS_NN2xy, [1], axis=-1)
            dS_NN2xx = tf.gather(tf.gradients(dS_NN2x, XY_it)[0], [0], axis=-1)
            dS_NN2yy = tf.gather(tf.gradients(dS_NN2y, XY_it)[0], [1], axis=-1)

            div_grad_NS = nu_s*(dN_NN2x*dS_NN2x+ N_NN*dS_NN2xx + dN_NN2y*dS_NN2y + N_NN*dS_NN2yy)
            temp2DS_NN = alpha*N_NN - (1-A/N_NN)*beta_i*S_NN*I_NN - (1-A/N_NN)*beta_e*S_NN*E_NN -mu*S_NN + div_grad_NS
            Loss2dSNNt = tf.reduce_mean(tf.square(dS_NN2t - temp2DS_NN))

            gradE_NN2xy = tf.gradients(E_NN, XY_it)[0]
            dE_NN2x = tf.gather(gradE_NN2xy, [0], axis=-1)
            dE_NN2y = tf.gather(gradE_NN2xy, [1], axis=-1)
            dE_NN2xx = tf.gather(tf.gradients(dE_NN2x, XY_it)[0], [0], axis=-1)
            dE_NN2yy = tf.gather(tf.gradients(dE_NN2y, XY_it)[0], [1], axis=-1)

            div_grad_NE = nu_e * (dN_NN2x * dE_NN2x + N_NN * dE_NN2xx + dN_NN2y * dE_NN2y + N_NN * dE_NN2yy)
            temp2DE_NN = (1 - A / N_NN) * beta_i * S_NN * I_NN - (1 - A / N_NN) * beta_e * S_NN * E_NN - sigma*E_NN \
                         - phi_e*E_NN - mu*E_NN + div_grad_NE
            Loss2dENNt = tf.reduce_mean(tf.square(dE_NN2t - temp2DE_NN))

            gradI_NN2xy = tf.gradients(I_NN, XY_it)[0]
            dI_NN2x = tf.gather(gradI_NN2xy, [0], axis=-1)
            dI_NN2y = tf.gather(gradI_NN2xy, [1], axis=-1)
            dI_NN2xx = tf.gather(tf.gradients(dI_NN2x, XY_it)[0], [0], axis=-1)
            dI_NN2yy = tf.gather(tf.gradients(dI_NN2y, XY_it)[0], [1], axis=-1)

            div_grad_NI = nu_i * (dN_NN2x * dI_NN2x + N_NN * dI_NN2xx + dN_NN2y * dI_NN2y + N_NN * dI_NN2yy)
            temp2DI_NN = sigma*E_NN - phi_d*I_NN - phi_r*I_NN - mu*I_NN + div_grad_NI
            Loss2dINNt = tf.reduce_mean(tf.square(dI_NN2t - temp2DI_NN))

            gradR_NN2xy = tf.gradients(R_NN, XY_it)[0]
            dR_NN2x = tf.gather(gradR_NN2xy, [0], axis=-1)
            dR_NN2y = tf.gather(gradR_NN2xy, [1], axis=-1)
            dR_NN2xx = tf.gather(tf.gradients(dR_NN2x, XY_it)[0], [0], axis=-1)
            dR_NN2yy = tf.gather(tf.gradients(dR_NN2y, XY_it)[0], [1], axis=-1)

            div_grad_NR = nu_i * (dN_NN2x * dR_NN2x + N_NN * dR_NN2xx + dN_NN2y * dR_NN2y + N_NN * dR_NN2yy)
            temp2DR_NN = phi_r*I_NN + phi_e*E_NN - mu * R_NN + div_grad_NR
            Loss2dRNNt = tf.reduce_mean(tf.square(dR_NN2t - temp2DR_NN))

            temp2DD_NN = phi_d*I_NN
            Loss2dDNNt = tf.reduce_mean(tf.square(dD_NN2t - temp2DD_NN))

            if R['regular_weight_model'] == 'L1':
                regular_WB2S = DNN_base.regular_weights_biases_L1(Weight2S, Bias2S)      # 正则化权重参数 L1正则化
                regular_WB2E = DNN_base.regular_weights_biases_L1(Weight2E, Bias2E)
                regular_WB2I = DNN_base.regular_weights_biases_L1(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L1(Weight2R, Bias2R)
                regular_WB2D = DNN_base.regular_weights_biases_L1(Weight2D, Bias2D)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2S = DNN_base.regular_weights_biases_L2(Weight2S, Bias2S)      # 正则化权重参数 L2正则化
                regular_WB2E = DNN_base.regular_weights_biases_L2(Weight2E, Bias2E)
                regular_WB2I = DNN_base.regular_weights_biases_L2(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L2(Weight2R, Bias2R)
                regular_WB2D = DNN_base.regular_weights_biases_L2(Weight2D, Bias2D)
            else:
                regular_WB2S = tf.constant(0.0)
                regular_WB2E = tf.constant(0.0)
                regular_WB2I = tf.constant(0.0)
                regular_WB2R = tf.constant(0.0)
                regular_WB2D = tf.constant(0.0)

            Loss2S = Loss2SNN + Loss2dSNNt + wb_penalty*regular_WB2S
            Loss2E = Loss2ENN + Loss2dENNt + wb_penalty*regular_WB2E
            Loss2I = Loss2INN + Loss2dINNt + wb_penalty*regular_WB2I
            Loss2R = Loss2RNN + Loss2dRNNt + wb_penalty*regular_WB2R
            Loss2D = Loss2DNN + Loss2dDNNt + wb_penalty*regular_WB2D
            Loss2N = tf.reduce_mean(tf.square(N_NN - N_true))

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_Loss2S = my_optimizer.minimize(Loss2S, global_step=global_steps)
            train_Loss2E = my_optimizer.minimize(Loss2E, global_step=global_steps)
            train_Loss2I = my_optimizer.minimize(Loss2I, global_step=global_steps)
            train_Loss2R = my_optimizer.minimize(Loss2R, global_step=global_steps)
            train_Loss2D = my_optimizer.minimize(Loss2D, global_step=global_steps)
            train_Loss2N = my_optimizer.minimize(Loss2N, global_step=global_steps)
            train_Loss = tf.group(train_Loss2S, train_Loss2E, train_Loss2I, train_Loss2R, train_Loss2D, train_Loss2N)

    t0 = time.time()
    loss_s_all, loss_e_all, loss_i_all, loss_r_all, loss_d_all, loss_n_all = [], [], [], [], [], []
    test_epoch = []
    test_mse2s_all, test_mse2e_all, test_mse2i_all, test_mse2r_all, test_mse2d_all = [], [], [], [], []
    test_rel2s_all, test_rel2e_all, test_rel2i_all, test_rel2r_all, test_rel2d_all = [], [], [], [], []

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            t_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xy_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            tmp_lr = tmp_lr * (1 - lr_decay)
            train_option = True
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            elif R['activate_stage_penalty'] == 2:
                if i_epoch < int(R['max_epoch'] / 3):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < 2*int(R['max_epoch'] / 3):
                    temp_penalty_bd = 10 * bd_penalty_init
                else:
                    temp_penalty_bd = 50 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            _, loss2s_tmp, loss2e_tmp, loss2i_tmp, loss2r_tmp, loss2d_tmp, loss2n_tmp = sess.run(
                [train_Loss, Loss2S, Loss2E, Loss2I, Loss2R, Loss2D, Loss2N],
                feed_dict={T_it:t_it_batch, XY_it: xy_it_batch, in_learning_rate: tmp_lr, train_opt: train_option,
                           bd_penalty: temp_penalty_bd})

            loss_s_all.append(loss2s_tmp)
            loss_e_all.append(loss2e_tmp)
            loss_i_all.append(loss2i_tmp)
            loss_r_all.append(loss2r_tmp)
            loss_d_all.append(loss2d_tmp)
            loss_n_all.append(loss2n_tmp)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, loss2s_tmp, loss2e_tmp,
                                    loss2i_tmp, loss2r_tmp, loss2d_tmp, loss2n_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                s_nn2test, e_nn2test, i_nn2test, r_nn2test, d_nn2test = sess.run(
                    [S_NN, E_NN, I_NN, R_NN, D_NN], feed_dict={
                        T_it: t_it_batch, XY_it: test_xy_bach, train_opt: train_option})
                s_true2test, e_true2test, i_true2test, r_true2test, d_true2test = sess.run(
                    [S_true, E_true, I_true, R_true, D_true], feed_dict={
                        T_it: t_it_batch, XY_it: test_xy_bach, train_opt: train_option})
                point_square_error2s = np.square(s_nn2test - s_true2test)
                point_square_error2e = np.square(e_nn2test - e_true2test)
                point_square_error2i = np.square(i_nn2test - i_true2test)
                point_square_error2r = np.square(r_nn2test - r_true2test)
                point_square_error2d = np.square(d_nn2test - d_true2test)
                mse2s = np.mean(point_square_error2s)
                mse2e = np.mean(point_square_error2e)
                mse2i = np.mean(point_square_error2i)
                mse2r = np.mean(point_square_error2r)
                mse2d = np.mean(point_square_error2d)
                test_mse2s_all.append(mse2s)
                test_mse2e_all.append(mse2e)
                test_mse2i_all.append(mse2i)
                test_mse2r_all.append(mse2r)
                test_mse2d_all.append(mse2d)

                res2s = mse2s / np.mean(np.square(s_true2test))
                res2e = mse2e / np.mean(np.square(e_true2test))
                res2i = mse2i / np.mean(np.square(i_true2test))
                res2r = mse2r / np.mean(np.square(r_true2test))
                res2d = mse2d / np.mean(np.square(d_true2test))
                test_rel2s_all.append(res2s)
                test_rel2e_all.append(res2e)
                test_rel2i_all.append(res2i)
                test_rel2r_all.append(res2r)
                test_rel2d_all.append(res2d)

                # print('mean square error of predict and real for testing: %10f' % mse2test)
                # print('residual error of predict and real for testing: %10f\n' % res2test)
                # DNN_tools.log_string('mean square error of predict and real for testing: %10f' % mse2test, log_fileout)
                # DNN_tools.log_string('residual error of predict and real for testing: %10f\n\n' % res2test, log_fileout)

        # -----------------------  save training result to mat file, then plot them ---------------------------------
        saveData.save_trainLoss2mat_1actFunc_Covid(
            loss_s_all, loss_e_all, loss_i_all, loss_r_all, loss_d_all, loss_n_all, actName=act_func, outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss_s', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_e_all, lossType='loss_e', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss_i', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss_r', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_d_all, lossType='loss_d', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_n_all, lossType='loss_n', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)

        # ------------------------------ save testing result to mat file, then plot them -------------------------------
        saveData.save_testMSE_REL2mat_1type(test_mse2s_all, test_rel2s_all, dataType='S', actName=act_func,
                                            outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat_1type(test_mse2e_all, test_rel2e_all, dataType='E', actName=act_func,
                                            outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat_1type(test_mse2i_all, test_rel2i_all, dataType='I', actName=act_func,
                                            outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat_1type(test_mse2r_all, test_rel2r_all, dataType='R', actName=act_func,
                                            outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat_1type(test_mse2d_all, test_rel2d_all, dataType='D', actName=act_func,
                                            outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'SEIRD2covid'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    R['eqs_name'] = 'SEIRD'
    R['input_dim'] = 1                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['batch_size'] = 5                   # 训练数据的批大小

    R['init_bd_penalty'] = 50             # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1       # 是否开启阶段调整边界惩罚项
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        R['init_bd_penalty'] = 1

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'
    # R['regular_weight'] = 0.000         # Regularization parameter for weights
    R['regular_weight'] = 0.001           # Regularization parameter for weights

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-4         # 学习率
        R['lr_decay'] = 5e-5              # 学习率 decay
    elif (20000 < R['max_epoch'] and 50000 >= R['max_epoch']):
        R['learning_rate'] = 1e-4         # 学习率
        R['lr_decay'] = 5e-5              # 学习率 decay
    else:
        R['learning_rate'] = 5e-5         # 学习率
        R['lr_decay'] = 1e-5              # 学习率 decay
    R['optimizer_name'] = 'Adam'          # 优化器
    # R['loss_function'] = 'L2_loss'
    R['loss_function'] = 'lncosh_loss'

    R['hidden_layers'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
    # R['hidden_layers'] = (80, 80, 60, 40, 40, 20)
    # R['hidden_layers'] = (100, 100, 80, 60, 60, 40)
    # R['hidden_layers'] = (200, 100, 100, 80, 50, 50)
    # R['hidden_layers'] = (300, 200, 200, 100, 80, 80)
    # R['hidden_layers'] = (400, 300, 300, 200, 100, 100)
    # R['hidden_layers'] = (500, 400, 300, 200, 200, 100, 100)
    # R['hidden_layers'] = (600, 400, 400, 300, 200, 200, 100)
    # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)

    # 网络模型的选择
    R['model'] = 'PDE_DNN'
    # R['model'] = 'PDE_DNN_BN'
    # R['model'] = 'PDE_DNN_scale'

    # 激活函数的选择
    # R['act_name'] = 'relu'
    R['act_name'] = 'tanh'
    # R['act_name'] = 'leaky_relu'
    # R['act_name'] = 'srelu'
    # R['act_name'] = 's2relu'
    # R['act_name'] = 'slrelu'
    # R['act_name'] = 'elu'
    # R['act_name'] = 'selu'
    # R['act_name'] = 'phi'

    solve_SEIRD2COVID(R)
