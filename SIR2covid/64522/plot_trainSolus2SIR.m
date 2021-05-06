clear all
close all
clc
data2itrain = load('i2train');
data2itrue = load('ITRUE2TRAIN');
itrain = double(data2itrain.i2train);
itrue = data2itrue.itrue2train;
format long

figure('name','i2train')
plot(itrain', 'c.-', 'linewidth', 2.0)
% set(gca,'yscale','log')
hold on

plot(itrue, 'm--', 'linewidth', 2.0)
hold on

lgi = legend({'i\_train','i\_true'},'Location', 'North','FontSize',18);

data2Strain = load('s2train');
Strain = data2Strain.s2train;
figure('name','s2train')
plot(Strain', 'r.-', 'linewidth', 2.0)
% set(gca,'yscale','log')
hold on
lgs=legend({'s\_train'},'Location', 'North','FontSize',18);


data2Rtrain = load('r2train');
Rtrain = data2Rtrain.r2train;
figure('name','r2train')
plot(Rtrain', 'b.-', 'linewidth', 2.0)
% set(gca,'yscale','log')
hold on
lgr=legend({'r\_train'},'Location', 'North','FontSize',18);





