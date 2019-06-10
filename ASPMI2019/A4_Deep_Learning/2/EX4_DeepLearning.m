%%% ASPMI Coursework PART 4: From LMS to Deep Learning
%%% PART 1-3
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed for random number generation

%% QUESTION 4: Nonlinear LMS
load time-series.mat;   %Load Data
N = length(y);          %Data Length

%Hyperparameters
mu = 1e-5;              %LMS Learning Rate
M = 4;                  %LMS AR Order
a = 60;                 %Activation fucntion scaling

%Variables initialisation
x = zeros(M,N);         %LMS input
X = zeros(M+1,N);       %LMS input (extended)
y_est = zeros(1,N);     %LMS predicted signal
w = zeros(M+1,N);       %LMS weights initialisation
net = zeros(1,N);       %Weight sum of LMS inputs
e = zeros(1, N);        %LMS Error Signal

for n = M+1:N
    for i = 1:M                         %Generate input u
        x(i, n) = y(n-i);               %Input
    end
    X(:,n) = [1 x(:,n)']';              %Extended input (with bias)
    net(n) = w(:,n)'*X(:,n);            %Net sum
    y_est(n) = a*tanh(net(n));          %Output
    e(n) = y(n) - y_est(n);             %Error
    w(:,n+1) = w(:,n) + mu*(sech(net(n))^2)*e(n)*X(:,n); %Weight update
end

E = 10*log10(e.^2);                     %Error power
mse = 10*log10(mean(e(M+1:end).^2));    %MSE
g = 10*log10((std(y_est(M+1:end))^2)/(std(e(M+1:end))^2));  %Prediction Gain

%PLOT RESULTS
subplot(2,3,1); scatter(y_est, y, '+'); grid on; grid minor; p2 = polyfit(y_est,y',1);
xlabel('$\hat{y}_{nl}$', 'Interpreter', 'Latex', 'FontSize', 16); ylabel('$y_0$', 'Interpreter', 'Latex', 'FontSize', 16);
title(['LMS Performance Evaluation: True signal against prediction (tanh activation function, a=', num2str(a), ')']);
legend({sprintf(strcat('Mean MSE: \t', num2str(round(mse,2)), ' dB \nMean Gain: \t', num2str(round(g,2)), ' dB'))});
subplot(2,1,2); plot(y, 'LineWidth', 1); hold on; plot(y_est, 'LineWidth', 1); grid on; grid minor;
xlabel('Time Steps (AU)', 'FontSize', 11); ylabel('Amplitude (AU)', 'FontSize', 12); title(['True signal and LMS Prediction against Time (a=', num2str(a), ')']);
legend({'$y_0$', '$\hat{y}_{nl}$'}, 'Interpreter', 'Latex'); 
subplot(2,3,2); plot(1:N,E); grid on; grid minor; xlabel('Time Steps (AU)', 'FontSize', 11); ylabel('Error Power (dB)', 'FontSize', 12);
title('Learning Curve for the Dynamical Perceptron with Bias');
for i = 1:length(w(:,1))
    subplot(2,3,3); plot(1:N, w(i,1:end-1), 'Linewidth', 1); hold on; grid on; grid minor;
    xlabel('Time Steps (AU)', 'FontSize', 11); ylabel('Amplitude (AU)', 'FontSize', 12);
    title('Weight Evolution for the Dynamical Perceptron with Bias');
end