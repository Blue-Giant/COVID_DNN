clear all
close all
clc
solus = load('solus2test');
format long
s_test = solus.snn2test;
i_test = solus.inn2test;
r_test = solus.rnn2test;

figure('name','s')
plot(s_test, 'b--','linewidth', 2.0)
hold on


figure('name','i')
plot(i_test, 'c.-', 'linewidth', 2.0)
set(gca,'yscale','log')
hold on

figure('name','r')
plot(r_test, 'm--', 'linewidth', 2.0)
hold on

paras = load('paras2test.mat');
testGamma = paras.gamma2test;
testBeta = paras.beta2test;

figure('name','beta')
plot(testBeta, 'm--',  'linewidth', 2.0)
title('\beta','Fontsize', 18)
hold on

figure('name','gamma')
plot(testGamma, 'm--', 'linewidth', 2.0)
title('\gamma', 'Fontsize', 18)
hold on


