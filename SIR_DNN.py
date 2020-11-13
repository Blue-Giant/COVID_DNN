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

    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_bd_penalty']), log_fileout)

    DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, temp_penalty_bd, penalty_wb2s, penalty_wb2i, penalty_wb2r,
                        loss_s, loss_i, loss_r, loss_n, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('boundary penalty: %f' % temp_penalty_bd)
    print('penalty weights and biases for S: %f' % penalty_wb2s)
    print('penalty weights and biases for I: %f' % penalty_wb2i)
    print('penalty weights and biases for R: %f' % penalty_wb2r)
    print('loss for S: %.10f' % loss_s)
    print('loss for I: %.10f' % loss_i)
    print('loss for R: %.10f' % loss_r)
    print('total loss: %.10f' % loss_n)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('boundary penalty: %f' % temp_penalty_bd, log_out)
    DNN_tools.log_string('penalty weights and biases for S: %f' % penalty_wb2s, log_out)
    DNN_tools.log_string('penalty weights and biases for I: %f' % penalty_wb2i, log_out)
    DNN_tools.log_string('penalty weights and biases for R: %f' % penalty_wb2r, log_out)
    DNN_tools.log_string('loss for S: %.10f' % loss_s, log_out)
    DNN_tools.log_string('loss for I: %.10f' % loss_i, log_out)
    DNN_tools.log_string('loss for R: %.10f' % loss_r, log_out)
    DNN_tools.log_string('total loss: %.10f' % loss_n, log_out)


def solve_SIR2COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    size2batch = R['batch_size']
    nn_true_penalty_init = R['init_bd_penalty']            # Regularization parameter for boundary conditions
    wb_penalty = R['regular_weight']                  # Regularization parameter for weights
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']
    act_func = R['act_name']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    flag2S = 'WB2S'
    flag2I = 'WB2I'
    flag2R = 'WB2R'
    hidden_layers = R['hidden_layers']
    Weight2S, Bias2S = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2S)
    Weight2I, Bias2I = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2I)
    Weight2R, Bias2R = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2R)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            T_it = tf.placeholder(tf.float32, name='T_it', shape=[None, out_dim])
            I_observe = tf.placeholder(tf.float32, name='I_observe', shape=[None, out_dim])
            N_observe = tf.placeholder(tf.float32, name='N_observe', shape=[None, out_dim])
            nn_true_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            in_beta = tf.placeholder_with_default(input=1e-5, shape=[], name='beta')
            in_gamma = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')
            beta = tf.exp(in_beta)
            gamma = tf.exp(in_gamma)
            if 'PDE_DNN' == str.upper(R['model']):
                S_NN = DNN_base.PDE_DNN(T_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN(T_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN(T_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func)
            elif 'PDE_DNN_BN' == str.upper(R['model']):
                S_NN = DNN_base.PDE_DNN_BN(T_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func, is_training=train_opt)
                I_NN = DNN_base.PDE_DNN_BN(T_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func, is_training=train_opt)
                R_NN = DNN_base.PDE_DNN_BN(T_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func, is_training=train_opt)
            elif 'PDE_DNN_SCALE' == str.upper(R['model']):
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                S_NN = DNN_base.PDE_DNN_scale(T_it, Weight2S, Bias2S, hidden_layers, freq, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN_scale(T_it, Weight2I, Bias2I, hidden_layers, freq, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN_scale(T_it, Weight2R, Bias2R, hidden_layers, freq, activate_name=act_func)

            N_NN = S_NN + I_NN + R_NN

            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]
            dN_NN2t = tf.gradients(N_NN, T_it)[0]

            temp_snn2t = -beta*S_NN*I_NN
            temp_inn2t = beta*S_NN*I_NN - gamma * I_NN
            temp_rnn2t = gamma *I_NN

            # LossS_Net_obs = tf.reduce_mean(tf.square(S_NN - S_observe))
            LossI_Net_obs = tf.reduce_mean(tf.square(I_NN - I_observe))
            # LossR_Net_obs = tf.reduce_mean(tf.square(R_NN - R_observe))
            LossN_Net_obs = tf.reduce_mean(tf.square(N_NN - N_observe))

            if R['regular_weight_model'] == 'L1':
                regular_WB2S = DNN_base.regular_weights_biases_L1(Weight2S, Bias2S)      # 正则化权重参数 L1正则化
                regular_WB2I = DNN_base.regular_weights_biases_L1(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L1(Weight2R, Bias2R)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2S = DNN_base.regular_weights_biases_L2(Weight2S, Bias2S)      # 正则化权重参数 L2正则化
                regular_WB2I = DNN_base.regular_weights_biases_L2(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L2(Weight2R, Bias2R)
            else:
                regular_WB2S = tf.constant(0.0)
                regular_WB2I = tf.constant(0.0)
                regular_WB2R = tf.constant(0.0)

            Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
            Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
            Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
            Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))

            PWB2S = wb_penalty*regular_WB2S
            PWB2I = wb_penalty*regular_WB2I
            PWB2R = wb_penalty*regular_WB2R

            Loss2S = Loss2dS + PWB2S
            Loss2I = nn_true_penalty*LossI_Net_obs + Loss2dI + PWB2I
            Loss2R = Loss2dR + PWB2R
            Loss2All = LossN_Net_obs + Loss2dN

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_Loss2S = my_optimizer.minimize(Loss2S, global_step=global_steps)
            train_Loss2I = my_optimizer.minimize(Loss2I, global_step=global_steps)
            train_Loss2R = my_optimizer.minimize(Loss2R, global_step=global_steps)
            train_Loss2All = my_optimizer.minimize(Loss2All, global_step=global_steps)
            train_Loss = tf.group(train_Loss2S, train_Loss2I, train_Loss2R, train_Loss2All)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_n_all = [], [], [], []

    filename = 'data2csv/Italia_data.csv'
    date, data = DNN_data.load_csvData(filename)
    ndata = np.ones(size2batch, dtype=np.float32)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(R['max_epoch'] + 1):
            t_batch, i_obs = DNN_data.randSample_Normalize_existData(date, data, batchsize=size2batch, normalFactor=10000)
            n_obs = ndata.reshape(size2batch, 1)
            tmp_lr = tmp_lr * (1 - lr_decay)
            train_option = True
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = nn_true_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * nn_true_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * nn_true_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * nn_true_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * nn_true_penalty_init
                else:
                    temp_penalty_bd = 500 * nn_true_penalty_init
            elif R['activate_stage_penalty'] == 2:
                if i_epoch < int(R['max_epoch'] / 3):
                    temp_penalty_bd = nn_true_penalty_init
                elif i_epoch < 2*int(R['max_epoch'] / 3):
                    temp_penalty_bd = 10 * nn_true_penalty_init
                else:
                    temp_penalty_bd = 50 * nn_true_penalty_init
            else:
                temp_penalty_bd = nn_true_penalty_init

            _, loss_s, loss_i, loss_r, loss_n, pwb2s, pwb2i, pwb2r = sess.run(
                [train_Loss, Loss2S, Loss2I, Loss2R, Loss2All, PWB2S, PWB2I, PWB2R],
                feed_dict={T_it: t_batch, I_observe: i_obs,
                           N_observe: n_obs, in_learning_rate: tmp_lr, train_opt: train_option,
                           nn_true_penalty: temp_penalty_bd})
            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_n_all.append(loss_n)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, pwb2s, pwb2i, pwb2r, loss_s,
                                    loss_i, loss_r, loss_n, log_out=log_fileout)


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'SIR2covid'
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

    R['eqs_name'] = 'SIR'
    R['input_dim'] = 1                   # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['batch_size'] = 5      # 训练数据的批大小

    R['init_bd_penalty'] = 50            # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1       # 是否开启阶段调整边界惩罚项
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        R['init_bd_penalty'] = 1

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'
    # R['regular_weight'] = 0.000           # Regularization parameter for weights
    R['regular_weight'] = 0.001         # Regularization parameter for weights

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-4  # 学习率
        R['lr_decay'] = 5e-5       # 学习率 decay
    elif (20000 < R['max_epoch'] and 50000 >= R['max_epoch']):
        R['learning_rate'] = 1e-4  # 学习率
        R['lr_decay'] = 5e-5       # 学习率 decay
    else:
        R['learning_rate'] = 5e-5  # 学习率
        R['lr_decay'] = 1e-5       # 学习率 decay
    R['optimizer_name'] = 'Adam'          # 优化器

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

    solve_SIR2COVID(R)
