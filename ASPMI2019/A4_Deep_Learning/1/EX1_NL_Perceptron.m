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
mu = 1e-5;              %LMS Learning Rate
M = 4;                  %LMS AR Order
R = 100;                %Number of realisations

%Variables initialisation
x = zeros(M,N);         %LMS Input signal
yh = zeros(1,N);        %LMS Output Signal
net = zeros(1,N);       %Weight sum of LMS inputs
YH = zeros(1,N);        %LMS Output (mean accross realisations)
e = zeros(1, N);        %LMS Error Signal
E = zeros(1,N);         %LMS Error (mean accross realisations)
mse = 0;                %MSE
g = 0;                  %Prediction gain

for r = 1:R                                 %Iterate over realisations
    w = randn(M,N);                         %LMS Filter Weights
    for n = M+1:N
        for i = 1:M                         %Generate input u
            x(i, n) = y(n-i);               %Filter input
        end
        yh(n) = w(:,n)'*x(:,n);             %Predictor output
        e(n) = y(n) - yh(n);                %Error
        w(:,n+1) = w(:,n) + mu*e(n)*x(:,n); %Weight update
    end
    E = E + 10*log10(e.^2);                 %Accumulate error power
    YH = YH + yh;                           %Accumulate LMS output
    mse(r) = 10*log10(mean(e(M+1:end).^2)); %MSE
    g(r) = 10*log10((std(yh(M+1:end))^2)/(std(e(M+1:end))^2));  %Prediction Gain
end
YH = YH/R;  %Mean output
E = E/R;    %Mean error

%PLOT RESULTS
figure(1); subplot(1,3,1); scatter(YH, y, '+'); grid on; grid minor; p1 = polyfit(YH,y',1);
xlabel('$\hat{y}$', 'Interpreter', 'Latex'); ylabel('$y_0$', 'Interpreter', 'Latex');
title('LMS Performance Evaluation: True signal against prediction (no activation function)');

subplot(1,3,2); histogram(mse, 'Normalization', 'probability', 'DisplayStyle', 'bar', 'FaceColor', [0.8500 0.3250 0.0980]);
grid on; grid minor; ylabel('Probability'); xlabel('MSE (dB)'); legend({['Mean MSE: ', num2str((round(mean(mse),2))), ' dB']});
title('Probability Distribution of the MSE across 200 realisations');
subplot(1,3,3); histogram(g, 'Normalization', 'probability', 'DisplayStyle', 'bar', 'FaceColor', [0.9290 0.6940 0.1250]);
grid on; grid minor; ylabel('Probability'); xlabel('Prediction Gain (dB)'); legend({['Mean Gain: ', num2str((round(mean(g),2))), ' dB']});
title('Probability Distribution of the Prediction Gain across 200 realisations');

%% QUESTION 2: Non-linear LMS
x = zeros(M,N);         %LMS Input signal
yh = zeros(1,N);        %LMS Output Signal
net = zeros(1,N);       %Weight sum of LMS inputs
mse = 0;                %MSE
g = 0;                  %Prediction gain
e = zeros(1, N);        %LMS Error Signal

a = [1 20 40 80, 200];                          %Scaling factors
figure(2);
for j = 1:length(a)                             %Iterate over scaling factors
    E = zeros(1,N);
    YH = zeros(1,N);
        w = zeros(M,N);                         %LMS Filter Weights
        for n = M+1:N
            for i = 1:M                         %Generate input u
                x(i, n) = y(n-i);               %Filter input
            end
            net(n) = w(:,n)'*x(:,n);            %Net weighted sum
            yh(n) = a(j)*tanh(net(n));          %Predictor output (with activation)
            e(n) = y(n) - yh(n);                %Error (filter output)
            w(:,n+1) = w(:,n) + mu*(sech(net(n))^2)*e(n)*x(:,n); %Weight update
        end
        E = 10*log10(e.^2);                     %Error power
        mse = 10*log10(mean(e(200:end).^2));    %MSE
        g = 10*log10((std(yh(200:end))^2)/(std(e(200:end))^2));  %Prediction gain
    
    %PLOT RESULTS
    subplot(2,length(a),j); scatter(yh, y, '+'); grid on; grid minor; p2 = polyfit(yh,y',1);
    xlabel('$\hat{y}_{nl}$', 'Interpreter', 'Latex'); ylabel('$y_0$', 'Interpreter', 'Latex');
    title(['LMS Performance Evaluation: True signal against prediction (tanh activation function, a=', num2str(a(j)), ')']);
    legend({sprintf(strcat('Mean MSE: \t', num2str(round(mse,2)), ' dB \nMean Gain: \t', num2str(round(g,2)), ' dB'))});
    subplot(2,length(a),j+length(a)); plot(y, 'LineWidth', 1); hold on; plot(yh, 'LineWidth', 1); grid on; grid minor;
    xlabel('Time Steps (AU)'); ylabel('Amplitude (AU)'); title(['True signal and LMS Prediction against Time (a=', num2str(a(j)), ')']);
    legend({'$y_0$', '$\hat{y}_{nl}$'}, 'Interpreter', 'Latex'); xlim([0 500]);
end
