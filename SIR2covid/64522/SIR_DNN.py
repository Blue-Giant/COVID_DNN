"""
@author: LXA
Benchmark Code of SIR model
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
import DNN
import plotData
import saveData
import DNN_LogPrint


def solve_SIR2COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_LogPrint.dictionary_out2file(R, log_fileout)

    log2trianSolus = open(os.path.join(log_out_path, 'train_Solus.txt'), 'w')      # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus = open(os.path.join(log_out_path, 'test_Solus.txt'), 'w')        # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus2 = open(os.path.join(log_out_path, 'test_Solus_temp.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    log2testParas = open(os.path.join(log_out_path, 'test_Paras.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    trainSet_szie = R['size2train']
    batchSize_train = R['batch_size2train']
    batchSize_test = R['batch_size2test']
    pt_penalty_init = R['init_penalty2predict_true']   # Regularization parameter for difference of predict and true
    wb_penalty = R['regular_weight']                   # Regularization parameter for weights
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']

    act_func2SIR = R['act2sir']
    act_func2paras = R['act2paras']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    flag2S = 'WB2S'
    flag2I = 'WB2I'
    flag2R = 'WB2R'
    flag2beta = 'WB2beta'
    flag2gamma = 'WB2gamma'
    hidden_sir = R['hidden2SIR']
    hidden_para = R['hidden2para']

    Weight2S, Bias2S = DNN.init_DNN(size2in=input_dim, size2out=out_dim, hiddens=hidden_sir, scope=flag2S,
                                    opt2Init=R['SIR_opt2init_NN'])
    Weight2I, Bias2I = DNN.init_DNN(size2in=input_dim, size2out=out_dim, hiddens=hidden_sir, scope=flag2I,
                                    opt2Init=R['SIR_opt2init_NN'])
    Weight2R, Bias2R = DNN.init_DNN(size2in=input_dim, size2out=out_dim, hiddens=hidden_sir, scope=flag2R,
                                    opt2Init=R['SIR_opt2init_NN'])

    Weight2beta, Bias2beta = DNN.init_DNN(size2in=input_dim, size2out=out_dim, hiddens=hidden_para, scope=flag2beta,
                                          opt2Init=R['Para_opt2init_NN'])
    Weight2gamma, Bias2gamma = DNN.init_DNN(size2in=input_dim, size2out=out_dim, hiddens=hidden_para, scope=flag2gamma,
                                            opt2Init=R['Para_opt2init_NN'])

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            T_it = tf.placeholder(tf.float32, name='T_it', shape=[None, input_dim])
            I_observe = tf.placeholder(tf.float32, name='I_observe', shape=[None, input_dim])
            N_observe = tf.placeholder(tf.float32, name='N_observe', shape=[None, input_dim])
            predict_true_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            SNN_temp = DNN.PDE_DNN(x=T_it, hiddenLayer=hidden_sir, Weigths=Weight2S, Biases=Bias2S,
                                   DNNmodel=R['model2sir'], activation=act_func2SIR, freqs=R['freqs'])
            INN_temp = DNN.PDE_DNN(x=T_it, hiddenLayer=hidden_sir, Weigths=Weight2I, Biases=Bias2I,
                                   DNNmodel=R['model2sir'], activation=act_func2SIR, freqs=R['freqs'])
            RNN_temp = DNN.PDE_DNN(x=T_it, hiddenLayer=hidden_sir, Weigths=Weight2R, Biases=Bias2R,
                                   DNNmodel=R['model2sir'], activation=act_func2SIR, freqs=R['freqs'])
            in_beta = DNN.PDE_DNN(x=T_it, hiddenLayer=hidden_para, Weigths=Weight2beta, Biases=Bias2beta,
                                  DNNmodel=R['model2paras'], activation=act_func2paras, freqs=R['freqs'])
            in_gamma = DNN.PDE_DNN(x=T_it, hiddenLayer=hidden_para, Weigths=Weight2gamma, Biases=Bias2gamma,
                                   DNNmodel=R['model2paras'], activation=act_func2paras, freqs=R['freqs'])

            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            if (R['total_population'] == R['scale_population']) and R['scale_population'] != 1:
                beta = tf.square(in_beta)
                gamma = tf.nn.sigmoid(in_gamma)
                # SNN = SNN_temp
                # INN = INN_temp
                # RNN = RNN_temp

                # SNN = tf.nn.relu(SNN_temp)
                # INN = tf.nn.relu(INN_temp)
                # RNN = tf.nn.relu(RNN_temp)

                # SNN = tf.abs(SNN_temp)
                # INN = tf.abs(INN_temp)
                # RNN = tf.abs(RNN_temp)

                # SNN = DNN_base.gauss(SNN_temp)
                # INN = tf.square(INN_temp)
                # RNN = tf.square(RNN_temp)

                # SNN = DNN_base.gauss(SNN_temp)
                # INN = tf.square(INN_temp)
                # RNN = tf.nn.sigmoid(RNN_temp)

                # SNN = DNN_base.gauss(SNN_temp)
                # INN = tf.nn.sigmoid(INN_temp)
                # RNN = tf.square(RNN_temp)

                # SNN = tf.sqrt(tf.square(SNN_temp))
                # INN = tf.sqrt(tf.square(INN_temp))
                # RNN = tf.sqrt(tf.square(RNN_temp))

                SNN = tf.nn.sigmoid(SNN_temp)
                INN = tf.nn.sigmoid(INN_temp)
                RNN = tf.nn.sigmoid(RNN_temp)
                # SNN = tf.tanh(SNN_temp)
                # INN = tf.tanh(INN_temp)
                # RNN = tf.tanh(RNN_temp)
            else:
                beta = tf.square(in_beta)
                gamma = tf.nn.sigmoid(in_gamma)

                # SNN = SNN_temp
                # INN = INN_temp
                # RNN = RNN_temp

                # SNN = tf.nn.relu(SNN_temp)
                # INN = tf.nn.relu(INN_temp)
                # RNN = tf.nn.relu(RNN_temp)

                SNN = tf.nn.sigmoid(SNN_temp)
                INN = tf.nn.sigmoid(INN_temp)
                RNN = tf.nn.sigmoid(RNN_temp)

                # SNN = tf.tanh(SNN_temp)
                # INN = tf.tanh(INN_temp)
                # RNN = tf.tanh(RNN_temp)

            N_NN = SNN + INN + RNN

            dSNN2t = tf.gradients(SNN, T_it)[0]
            dINN2t = tf.gradients(INN, T_it)[0]
            dRNN2t = tf.gradients(RNN, T_it)[0]
            dN_NN2t = tf.gradients(N_NN, T_it)[0]

            temp_snn2t = -beta*SNN*INN
            temp_inn2t = beta*SNN*INN - gamma * INN
            temp_rnn2t = gamma * INN

            if str.lower(R['loss_function']) == 'l2_loss'and R['scale_up'] == 0:
                # LossS_Net_obs = tf.reduce_mean(tf.square(SNN - S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(INN - I_observe))
                # LossR_Net_obs = tf.reduce_mean(tf.square(RNN - R_observe))
                LossN_Net_obs = tf.reduce_mean(tf.square(N_NN - N_observe))

                Loss2dS = tf.reduce_mean(tf.square(dSNN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dINN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dRNN2t - temp_rnn2t))
                Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))
            elif str.lower(R['loss_function']) == 'l2_loss' and R['scale_up'] == 1:
                scale_up = R['scale_factor']
                # LossS_Net_obs = tf.reduce_mean(tf.square(scale_up*SNN - scale_up*S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(scale_up*INN - scale_up*I_observe))
                # LossR_Net_obs = tf.reduce_mean(tf.square(scale_up*RNN - scale_up*R_observe))
                LossN_Net_obs = tf.reduce_mean(tf.square(scale_up*N_NN - scale_up*N_observe))

                Loss2dS = tf.reduce_mean(tf.square(scale_up*dSNN2t - scale_up*temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(scale_up*dINN2t - scale_up*temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(scale_up*dRNN2t - scale_up*temp_rnn2t))
                Loss2dN = tf.reduce_mean(tf.square(scale_up*dN_NN2t))
            elif str.lower(R['loss_function']) == 'lncosh_loss':
                # LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(SNN - S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(INN - I_observe)))
                # LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(RNN - R_observe)))
                LossN_Net_obs = tf.reduce_mean(tf.log(tf.cosh(N_NN - N_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dSNN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dINN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dRNN2t - temp_rnn2t)))
                Loss2dN = tf.reduce_mean(tf.log(tf.cosh(dN_NN2t)))

            if R['regular_weight_model'] == 'L1':
                regular_WB2S = DNN_base.regular_weights_biases_L1(Weight2S, Bias2S)
                regular_WB2I = DNN_base.regular_weights_biases_L1(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L1(Weight2R, Bias2R)
                regular_WB2Beta = DNN_base.regular_weights_biases_L1(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L1(Weight2gamma, Bias2gamma)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2S = DNN_base.regular_weights_biases_L2(Weight2S, Bias2S)
                regular_WB2I = DNN_base.regular_weights_biases_L2(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L2(Weight2R, Bias2R)
                regular_WB2Beta = DNN_base.regular_weights_biases_L2(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L2(Weight2gamma, Bias2gamma)
            else:
                regular_WB2S = tf.constant(0.0)
                regular_WB2I = tf.constant(0.0)
                regular_WB2R = tf.constant(0.0)
                regular_WB2Beta = tf.constant(0.0)
                regular_WB2Gamma = tf.constant(0.0)

            PWB2S = wb_penalty*regular_WB2S
            PWB2I = wb_penalty*regular_WB2I
            PWB2R = wb_penalty*regular_WB2R
            PWB2Beta = wb_penalty * regular_WB2Beta
            PWB2Gamma = wb_penalty * regular_WB2Gamma

            Loss2S = Loss2dS + PWB2S
            Loss2I = predict_true_penalty * LossI_Net_obs + Loss2dI + PWB2I
            Loss2R = Loss2dR + PWB2R
            Loss2N = predict_true_penalty * LossN_Net_obs + Loss2dN
            Loss = Loss2S + Loss2I + Loss2R + Loss2N + PWB2Beta + PWB2Gamma

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 'train_group':
                train_Loss2S = my_optimizer.minimize(Loss2S, global_step=global_steps)
                train_Loss2I = my_optimizer.minimize(Loss2I, global_step=global_steps)
                train_Loss2R = my_optimizer.minimize(Loss2R, global_step=global_steps)
                train_Loss2N = my_optimizer.minimize(Loss2N, global_step=global_steps)
                train_Loss = my_optimizer.minimize(Loss, global_step=global_steps)
                train_Losses = tf.group(train_Loss2S, train_Loss2I, train_Loss2R, train_Loss2N, train_Loss)
            elif R['train_model'] == 'train_union_loss':
                train_Losses = my_optimizer.minimize(Loss, global_step=global_steps)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_n_all, loss_all = [], [], [], [], []
    test_epoch = []
    test_mse2I_all, test_rel2I_all = [], []

    # filename = 'data2csv/Wuhan.csv'
    # filename = 'data2csv/Italia_data.csv'
    filename = 'data2csv/Korea_data.csv'
    date, data = DNN_data.load_csvData(filename)

    assert(trainSet_szie + batchSize_test <= len(data))
    train_date, train_data2i, test_date, test_data2i = \
        DNN_data.split_csvData2train_test(date, data, size2train=trainSet_szie, normalFactor=R['scale_population'])

    if R['scale_population'] == 1:
        nbatch2train = np.ones(batchSize_train, dtype=np.float32)*float(R['total_population'])
    elif (R['total_population'] != R['scale_population']) and R['scale_population'] != 1:
        nbatch2train = np.ones(batchSize_train, dtype=np.float32) * (float(R['total_population'])/float(R['scale_population']))
    elif (R['total_population'] == R['scale_population']) and R['scale_population'] != 1:
        nbatch2train = np.ones(batchSize_train, dtype=np.float32)

    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
    test_t_bach = DNN_data.sample_testDays_serially(test_date, batchSize_test)
    i_obs_test = DNN_data.sample_testData_serially(test_data2i, batchSize_test, normalFactor=1.0)
    print('The test data about i:\n', str(np.transpose(i_obs_test)))
    print('\n')
    DNN_tools.log_string('The test data about i:\n%s\n' % str(np.transpose(i_obs_test)), log_fileout)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(R['max_epoch'] + 1):
            t_batch, i_obs = \
                DNN_data.randSample_Normalize_existData(train_date, train_data2i, batchsize=batchSize_train,
                                                        normalFactor=1.0, sampling_opt=R['opt2sample'])
            n_obs = nbatch2train.reshape(batchSize_train, 1)
            tmp_lr = tmp_lr * (1 - lr_decay)
            train_option = True
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_pt = 10 * pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_pt = 50 * pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_pt = 100 * pt_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_pt = 200 * pt_penalty_init
                else:
                    temp_penalty_pt = 500 * pt_penalty_init
            elif R['activate_stage_penalty'] == 2:
                if i_epoch < int(R['max_epoch'] / 3):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < 2*int(R['max_epoch'] / 3):
                    temp_penalty_pt = 10 * pt_penalty_init
                else:
                    temp_penalty_pt = 50 * pt_penalty_init
            else:
                temp_penalty_pt = pt_penalty_init

            _, loss_s, loss_i, loss_r, loss_n, loss, pwb2s, pwb2i, pwb2r = sess.run(
                [train_Losses, Loss2S, Loss2I, Loss2R, Loss2N, Loss, PWB2S, PWB2I, PWB2R],
                feed_dict={T_it: t_batch, I_observe: i_obs, N_observe: n_obs, in_learning_rate: tmp_lr,
                           train_opt: train_option, predict_true_penalty: temp_penalty_pt})

            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_n_all.append(loss_n)
            loss_all.append(loss)

            if i_epoch % 1000 == 0:
                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的训练结果
                DNN_LogPrint.print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_pt, pwb2s, pwb2i,
                                                 pwb2r, loss_s, loss_i, loss_r, loss_n, loss, log_out=log_fileout)

                s_nn2train, i_nn2train, r_nn2train = sess.run(
                    [SNN, INN, RNN], feed_dict={T_it: np.reshape(train_date, [-1, 1])})

                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的测试结果
                test_epoch.append(i_epoch / 1000)
                train_option = False
                s_nn2test, i_nn2test, r_nn2test, beta_test, gamma_test = sess.run(
                    [SNN, INN, RNN, beta, gamma], feed_dict={T_it: test_t_bach, train_opt: train_option})
                point_ERR2I = np.square(i_nn2test - i_obs_test)
                test_mse2I = np.mean(point_ERR2I)
                test_mse2I_all.append(test_mse2I)
                test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
                test_rel2I_all.append(test_rel2I)

                DNN_tools.print_and_log_test_one_epoch(test_mse2I, test_rel2I, log_out=log_fileout)
                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch), log2testSolus)
                DNN_tools.log_string('The test result for s:\n%s\n' % str(np.transpose(s_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for i:\n%s\n' % str(np.transpose(i_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for r:\n%s\n\n' % str(np.transpose(r_nn2test)), log2testSolus)

                # --------以下代码为输出训练过程中 S_NN_temp, I_NN_temp, R_NN_temp, in_beta, in_gamma 的测试结果-------------
                s_nn_temp2test, i_nn_temp2test, r_nn_temp2test, in_beta_test, in_gamma_test = sess.run(
                    [SNN_temp, INN_temp, RNN_temp, in_beta, in_gamma],
                    feed_dict={T_it: test_t_bach, train_opt: train_option})

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch), log2testSolus2)
                DNN_tools.log_string('The test result for s_temp:\n%s\n' % str(np.transpose(s_nn_temp2test)), log2testSolus2)
                DNN_tools.log_string('The test result for i_temp:\n%s\n' % str(np.transpose(i_nn_temp2test)), log2testSolus2)
                DNN_tools.log_string('The test result for r_temp:\n%s\n\n' % str(np.transpose(r_nn_temp2test)), log2testSolus2)

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch), log2testParas)
                DNN_tools.log_string('The test result for in_beta:\n%s\n' % str(np.transpose(in_beta_test)), log2testParas)
                DNN_tools.log_string('The test result for in_gamma:\n%s\n' % str(np.transpose(in_gamma_test)), log2testParas)

        DNN_tools.log_string('The train result for S:\n%s\n' % str(np.transpose(s_nn2train)), log2trianSolus)
        DNN_tools.log_string('The train result for I:\n%s\n' % str(np.transpose(i_nn2train)), log2trianSolus)
        DNN_tools.log_string('The train result for R:\n%s\n\n' % str(np.transpose(r_nn2train)), log2trianSolus)

        saveData.true_value2convid(train_data2i, name2Array='itrue2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(s_nn2train, name2solus='s2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(i_nn2train, name2solus='i2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(r_nn2train, name2solus='r2train', outPath=R['FolderName'])

        saveData.save_SIR_trainLoss2mat_Covid(loss_s_all, loss_i_all, loss_r_all, loss_n_all, actName=act_func2SIR,
                                              outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_n_all, lossType='loss2n', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)

        saveData.true_value2convid(i_obs_test, name2Array='i_true2test', outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse2I_all, test_rel2I_all, actName='Infected', outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse2I_all, test_rel2I_all, test_epoch, actName='Infected', seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)
        saveData.save_SIR_testSolus2mat_Covid(s_nn2test, i_nn2test, r_nn2test, name2solus1='snn2test',
                                              name2solus2='inn2test', name2solus3='rnn2test', outPath=R['FolderName'])
        saveData.save_SIR_testParas2mat_Covid(beta_test, gamma_test, name2para1='beta2test', name2para2='gamma2test',
                                              outPath=R['FolderName'])

        plotData.plot_testSolu2convid(i_obs_test, name2solu='i_true', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(s_nn2test, name2solu='s_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(i_nn2test, name2solu='i_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(r_nn2test, name2solu='r_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])

        plotData.plot_testSolus2convid(i_obs_test, i_nn2test, name2solu1='i_true', name2solu2='i_test',
                                       coord_points2test=test_t_bach, seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plot_testSolu2convid(beta_test, name2solu='beta_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(gamma_test, name2solu='gamma_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])


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

    # ----------------------------------------- Convid 设置 ---------------------------------
    R['eqs_name'] = 'SIR'
    R['input_dim'] = 1                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数
    R['total_population'] = 9776000

    R['scale_population'] = 9776000
    # R['scale_population'] = 100000
    # R['scale_population'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['size2train'] = 70                    # 训练集的大小
    R['batch_size2train'] = 20              # 训练数据的批大小
    R['batch_size2test'] = 10               # 训练数据的批大小
    # R['opt2sample'] = 'random_sample'     # 训练集的选取方式--随机采样
    R['opt2sample'] = 'rand_sample_sort'    # 训练集的选取方式--随机采样后按时间排序

    R['init_penalty2predict_true'] = 50   # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1       # 是否开启阶段调整边界惩罚项
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        # R['init_penalty2predict_true'] = 1000
        # R['init_penalty2predict_true'] = 100
        # R['init_penalty2predict_true'] = 50
        # R['init_penalty2predict_true'] = 20
        R['init_penalty2predict_true'] = 1

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight'] = 0.000             # Regularization parameter for weights

    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'        # The model of regular weights and biases
    # R['regular_weight'] = 0.001           # Regularization parameter for weights
    # R['regular_weight'] = 0.0005          # Regularization parameter for weights
    # R['regular_weight'] = 0.0001            # Regularization parameter for weights
    R['regular_weight'] = 0.00005          # Regularization parameter for weights
    # R['regular_weight'] = 0.00001        # Regularization parameter for weights

    R['optimizer_name'] = 'Adam'           # 优化器
    R['loss_function'] = 'L2_loss'
    R['scale_up'] = 1
    R['scale_factor'] = 100
    # R['loss_function'] = 'lncosh_loss'
    # R['train_model'] = 'train_group'
    R['train_model'] = 'train_union_loss'

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-3           # 学习率
        R['lr_decay'] = 1e-4                # 学习率 decay
        # R['learning_rate'] = 2e-4         # 学习率
        # R['lr_decay'] = 5e-5              # 学习率 decay
    elif (20000 < R['max_epoch'] and 50000 >= R['max_epoch']):
        # R['learning_rate'] = 1e-3         # 学习率
        # R['lr_decay'] = 1e-4              # 学习率 decay
        # R['learning_rate'] = 2e-4         # 学习率
        # R['lr_decay'] = 1e-4              # 学习率 decay
        R['learning_rate'] = 1e-4         # 学习率
        R['lr_decay'] = 5e-5              # 学习率 decay
    else:
        R['learning_rate'] = 5e-5         # 学习率
        R['lr_decay'] = 1e-5              # 学习率 decay

    # 网络模型的选择
    R['model2sir'] = 'DNN'
    # R['model2sir'] = 'DNN_modify'
    # R['model2sir'] = 'DNN_scale'
    # R['model2sir'] = 'DNN_scaleOut'
    # R['model2sir'] = 'DNN_FourierBase'

    R['model2paras'] = 'DNN'
    # R['model2paras'] = 'DNN_modify'
    # R['model2paras'] = 'DNN_scale'
    # R['model2paras'] = 'DNN_scaleOut'
    # R['model2paras'] = 'DNN_FourierBase'

    if R['model2sir'] == 'DNN_FourierBase':
        R['hidden2SIR'] = (35, 50, 30, 30, 20)          # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # R['hidden2SIR'] = (10, 10, 8, 6, 6, 3)        # it is used to debug our work
        R['hidden2SIR'] = (70, 50, 30, 30, 20)          # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # R['hidden2SIR'] = (80, 80, 60, 40, 40, 20)    # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # R['hidden2SIR'] = (100, 100, 80, 60, 60, 40)
        # R['hidden2SIR'] = (200, 100, 100, 80, 50, 50)

    if R['model2paras'] == 'DNN_FourierBase':
        R['hidden2para'] = (35, 50, 30, 30, 20)         # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # R['hidden2para'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
        R['hidden2para'] = (70, 50, 30, 30, 20)         # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # R['hidden2para'] = (80, 80, 60, 40, 40, 20)   # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # R['hidden2para'] = (100, 100, 80, 60, 60, 40)
        # R['hidden2para'] = (200, 100, 100, 80, 50, 50)

    # R['SIR_opt2init_NN'] = 'General_Init'
    # R['SIR_opt2init_NN'] = 'Truncated_normal_init'
    R['SIR_opt2init_NN'] = 'Xavier_init'
    R['freqs'] = [1]
    if R['model2sir'] == 'DNN_FourierBase':
        R['SIR_opt2init_NN'] = 'Xavier_init_Fourier'
        R['freqs'] = np.concatenate(([1, 1, 1, 1, 1], np.arange(1, 26)), axis=0)

    # R['Para_opt2init_NN'] = 'General_Init'
    # R['Para_opt2init_NN'] = 'Truncated_normal_init'
    R['Para_opt2init_NN'] = 'Xavier_init'
    if R['model2sir'] == 'DNN_FourierBase':
        R['Para_opt2init_NN'] = 'Xavier_init_Fourier'

    # 激活函数的选择
    # R['act2sir'] = 'relu'
    R['act2sir'] = 'tanh'                 # 这个激活函数比较s2ReLU合适
    # R['act2sir'] = 'sigmod'
    # R['act2sir'] = 'leaky_relu'
    # R['act2sir'] = 'srelu'
    # R['act2sir'] = 's2relu'
    # R['act2sir'] = 'elu'
    # R['act2sir'] = 'selu'
    # R['act2sir'] = 'phi'

    # R['act2paras'] = 'relu'
    R['act2paras'] = 'tanh'                 # 这个激活函数比较s2ReLU合适
    # R['act2paras'] = 'sigmod'
    # R['act2paras'] = 'leaky_relu'
    # R['act2paras'] = 'srelu'
    # R['act2paras'] = 's2relu'
    # R['act2paras'] = 'elu'
    # R['act2paras'] = 'selu'
    # R['act2paras'] = 'phi'

    solve_SIR2COVID(R)
