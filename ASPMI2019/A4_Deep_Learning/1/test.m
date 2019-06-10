%%% ASPMI Coursework PART 4: From LMS to Deep Learning
%%% PART 1-3
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed for random number generation

%% QUESTION 1: Linear LMS
load time-series.mat;   %Load Data
y = y - mean(y);        %Remove Mean
N = length(y);          %Data Length

%Data standardisation (optionnal)
% y = 2*((y-min(y))/(max(y)-min(y)))-1;

%Hyperparameters
mu = [1e-5 5e-4];       %LMS Learning Rate
M = 4;                  %LMS AR Order
R = 100;

%Variables initialisation
x = zeros(M,N);         %LMS Input signal
yh = zeros(1,N);        %LMS Output Signal
YH = zeros(1,N);        %LMS Output (mean accross realisations)
e = zeros(1, N);        %LMS Error Signal
E = zeros(1,N);         %LMS Error (mean accross realisations)
mse = 0;                %MSE
g = 0;                  %Prediction gain

for s = 1:length(mu)
for r = 1:R                                 %Iterate over realisations
    w = randn(M,N);                         %LMS Filter Weights
    for n = M+1:N
        for i = 1:M                         %Generate input u
            x(i, n) = y(n-i);               %Filter input
        end
        yh(n) = w(:,n)'*x(:,n);             %Predictor output
        e(n) = y(n) - yh(n);                %Error
        w(:,n+1) = w(:,n) + mu(s)*e(n)*x(:,n); %Weight update
    end
    E = E + 10*log10(e.^2);                 %Accumulate error power
    YH = YH + yh;                           %Accumulate LMS output
    mse(r) = 10*log10(mean(e(M+1:end).^2)); %MSE
    g(r) = 10*log10((std(yh(M+1:end))^2)/(std(e(M+1:end))^2));  %Prediction Gain
end
YH = YH/R;  %Mean output
E = E/R;    %Mean error
subplot(1,2,s); plot(y, 'Linewidth', 1);  hold on; plot(YH, 'Linewidth', 1);
ylabel('$\hat{y}$', 'Interpreter', 'Latex'); xlabel('Time index (AU)');
title('LMS Performance Evaluation: True signal against prediction (no activation function)');
xlim([1 500]); grid on; grid minor;
end
