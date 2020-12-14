clear all
close all
clc
solus = load('solus2test');
format long
s_test = solus.snn2test;
e_test = solus.enn2test;
i_test = solus.inn2test;
r_test = solus.rnn2test;

figure('name','s')
plot(s_test, 'b--')
hold on

figure('name','e')
plot(e_test, 'r--')
hold on

figure('name','i')
plot(i_test, 'c.-')
set(gca,'yscale','log')
hold on

figure('name','r')
plot(r_test, 'm--')
hold on


