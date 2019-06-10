%%% ASPMI Coursework PART 4: From LMS to Deep Learning
%%% PART 1-3
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed for random number generation

%% QUESTION 5: Training Weights
load time-series.mat;   %Load Data
L = length(y);          %Data Length

%Hyperparameters
mu = 1e-5;              %LMS Learning Rate
M = 4;                  %LMS AR Order
a = 60;                 %Activation fucntion scaling

initial = 20;           %Overfit model to the first "initial" steps
epochs = 100;           %Number of epochs for initial weigths training
N = epochs*(initial-M); %Total number of iterations for weights training

%Variables initialisation
x = zeros(M,N);         %LMS input
X = zeros(M+1,N);       %LMS input (extended)
y_est = zeros(1,N);     %LMS predicted signal
w = zeros(M+1,N);       %LMS weights initialisation
net = zeros(1,N);       %Weight sum of LMS inputs
e = zeros(1,N);         %LMS Error Signal
err = zeros(1,epochs);
weight = zeros(M+1,epochs);

n = 0;
for i = 1:epochs                            %Repeat epoch times
    for j = M+1:initial                     %Fit model to the first "initial" samples
        n = n +1;                           %Increment n
        for k = 1:M                         %Generate input
            x(k, n) = y(j-k);               %Input
        end
        X(:,n) = [1 x(:,n)']';              %Extended input (with bias)
        net(n) = w(:,n)'*X(:,n);            %Net sum
        y_est(n) = a*tanh(net(n));          %Output
        e(n) = y(j) - y_est(n);             %Error
        w(:,n+1) = w(:,n) + mu*(sech(net(n))^2)*e(n)*X(:,n); %Weight update
    end
    err(i) = 10*log10(e(n)^2);
    weight(:,i) = w(:,n);
end
w_0 = w(:,n+1);                         %Initial weights
E = 10*log10(e.^2);                     %Error power
mse = 10*log10(mean(e(M+1:end).^2));    %MSE
g = 10*log10((std(y_est(M+1:end))^2)/(std(e(M+1:end))^2));  %Prediction Gain

%PLOT RESULTS
figure;
subplot(1,2,1); plot(1:epochs, err, 'Linewidth', 2); grid on; grid minor; xlabel('Epochs', 'FontSize', 11); ylabel('Error Power (dB)', 'FontSize', 12);
title('Learning Curve for pre-training the Dynamical Perceptron Weight');
for i = 1:length(w(:,1))
    subplot(1,2,2); plot(1:epochs, weight(i,:), 'Linewidth', 1); hold on; grid on; grid minor;
    xlabel('Epochs', 'FontSize', 11); ylabel('Amplitude (AU)', 'FontSize', 12);
    title('Weight Evolution during pre-training of the Dynamical Perceptron Weights'); xlim([1 epochs]);
end

%% QUESTION 5: Use Trained Weights
%Variables initialisation
x = zeros(M,L);         %LMS input
X = zeros(M+1,L);       %LMS input (extended)
y_est = zeros(1,L);     %LMS predicted signal
net = zeros(1,L);       %Weight sum of LMS inputs
e = zeros(1,L);         %LMS Error Signal
w = zeros(M+1,L);       %LMS weights initialisation
for i = 1:M+1
    w(:,i) = w_0;       %Use trained weigths
end

for n = M+1:L
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
figure;
subplot(2,3,1); scatter(y_est, y, '+'); grid on; grid minor; p2 = polyfit(y_est,y',1);
xlabel('$\hat{y}_{nl}$', 'Interpreter', 'Latex', 'FontSize', 16); ylabel('$y_0$', 'Interpreter', 'Latex', 'FontSize', 16);
title(['Dynamical Perceptron Performance Evaluation: True signal against prediction (tanh activation function, a=', num2str(a), ')']);
legend({sprintf(strcat('Mean MSE: \t', num2str(round(mse,2)), ' dB \nMean Gain: \t', num2str(round(g,2)), ' dB'))});
subplot(2,1,2); plot(y, 'LineWidth', 1); hold on; plot(y_est, 'LineWidth', 1); grid on; grid minor;
xlabel('Time Steps (AU)', 'FontSize', 11); ylabel('Amplitude (AU)', 'FontSize', 12); title(['True signal and LMS Prediction against Time using w_{init} (a=', num2str(a), ')']);
legend({'$y_0$', '$\hat{y}_{nl}$'}, 'Interpreter', 'Latex');
subplot(2,3,2); plot(1:L,E); grid on; grid minor; xlabel('Time Steps (AU)', 'FontSize', 11); ylabel('Error Power (dB)', 'FontSize', 12);
title('Learning Curve for the Dynamical Perceptron with Bias using w_{init}');
for i = 1:length(w(:,1))
    subplot(2,3,3); plot(1:L, w(i,1:end-1), 'Linewidth', 1); hold on; grid on; grid minor;
    xlabel('Time Steps (AU)', 'FontSize', 11); ylabel('Amplitude (AU)', 'FontSize', 12);
    title('Weight Evolution for the Dynamical Perceptron with Bias using w_{init}');
end