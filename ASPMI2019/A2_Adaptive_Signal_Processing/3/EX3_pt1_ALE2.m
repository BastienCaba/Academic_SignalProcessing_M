%%% ASPMI Coursework PART 2: Adaptive Signal Processing
%%% 2.3 Adaptive Noise Cancelation
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed for random number generation

%% TASK B: Hyperparameters of ALE
%PARAMETERS
fs = 1;             %Sampling frequency (Hz)
N = 1000;           %Signal length
t = (0:N-1)/fs;     %Time axis (s)
A = 1;              %Sinusoid amplitude
w_0 = 0.01*pi;      %Sinusoid angular frequency (rad/s)
f_0 = w_0/(2*pi);   %Sinusoid frequency (Hz)
R = 50;             %Number of realisations

D = 25;             %Delay delta
M = 5;              %Filter order
mspe = 0;           %Initialise MSPE
x_hat_sum = zeros(1,N); %Initialise x_hat signal

figure;
for i = 1:R
    e = zeros(length(D),N);     %Initialise error
    x_hat = zeros(length(D),N); %Initialise signal estimate
    
    x = A*sin(w_0*t);   %Clean sinusoid signal
    v = randn(1,N);     %WGN
    eta = randn(1,N);   %Coloured corrupting noise
    
    for n = 3:N
        eta(n) = v(n) + 0.5*v(n-2);
    end
    s = x + eta;        %Noise-corrupted sinusoid
    
    [e, x_hat] = fALE(s, D, M, 0.01);               %ALE filtering
    mspe = mspe + mean((x(M:end)-x_hat(M:end)).^2); %MSPE
    x_hat_sum = x_hat_sum + x_hat;                  %Filter output accumulated
    
    %PLOT RESULTS
    subplot(1,2,1); hold on; hAx = gca; set(hAx,'xminorgrid','on','yminorgrid','on');
    p1 = plot(s, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1); uistack(p1,'bottom');
    p2 = plot(x_hat, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1); p2.Color(4) = 0.3;
    p3 = plot(x, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2); uistack(p3,'top');
    xlabel('Time Steps (AU)'); ylabel('Amplitude (AU)');
    title(['ALE Evaluation with M=', num2str(M), ' and \Delta=', num2str(D)]);
    [h, ~] = legend([p1, p2, p3], {['$s(n)$'], ['$\hat{x}(n)$'], ['$x(n)$']}, 'Interpreter', 'Latex');
end
mspe = 10*log10(mspe/R);    %Average MSPE across realisations
x_hat_sum = x_hat_sum/R;    %Average filter output across realisations

subplot(1,2,2); plot(x, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2); hold on;
plot(x_hat_sum, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1); grid on; grid minor; 
xlabel('Time Steps (AU)'); ylabel('Amplitude (AU)');
title(['ALE Performance averaged across 50 realisations with M=', num2str(M), ' and \Delta=', num2str(D)]);