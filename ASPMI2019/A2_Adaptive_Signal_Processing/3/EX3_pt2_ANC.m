%%% ASPMI Coursework 2019: ASSIGNMENT 2
%%% 2.3 Adaptive Noise Cancelation
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed for random number generation

%% TASK C: Explore ALE and ANC
%PARAMETERS
fs = 1;             %Sampling frequency (Hz)
N = 1000;           %Signal length
t = (0:N-1)/fs;     %Time axis (s)
A = 1;              %Sinusoid amplitude
w_0 = 0.01*pi;      %Sinusoid angular frequency (rad/s)
f_0 = w_0/(2*pi);   %Sinusoid frequency (Hz)
R = 50;             %Number of realisations

D = 3;              %Delay delta
M_ale = 3;          %ALE Filter order
M_anc = 3;          %ANC Filter order
mu_ale = 0.01;      %Learning rate ALE
mu_anc = 0.01;      %Learning rate ANC

mspe = zeros(1,2);  %Initialise MSPE
X_ale = zeros(1,N); %Initialise mean ALE output
X_anc = zeros(1,N); %Initialise mean ANC output

figure(1);
for i = 1:R
    x = A*sin(w_0*t);   %Clean sinusoid signal
    v = randn(1,N);     %WGN
    eta = randn(1,N);   %Coloured corrupting noise
    
    for n = 3:N
        eta(n) = v(n) + 0.5*v(n-2);
    end
    s = x + eta;        %Noise-corrupted sinusoid
    
    [e_ale, x_hat_ale] = fALE(s, D, M_ale, mu_ale);     %ALE
    [x_hat_anc, out_anc] = fANC(s, v, mu_anc, M_anc);   %ANC
    
    %Mean Squared Prediction Error
    mspe(1) = mspe(1) + mean((x(M_ale:end)-x_hat_ale(M_ale:end)).^2);
    mspe(2) = mspe(2) + mean((x(M_anc:end)-x_hat_anc(M_anc:end)).^2);
    
    X_ale = X_ale + x_hat_ale;  %ALE estimate
    X_anc = X_anc + x_hat_anc;  %ANC estimate
    
    %PLOT RESULTS
    subplot(1,2,1); hold on; hAx = gca; set(hAx,'xminorgrid','on','yminorgrid','on');
    p1 = plot(s, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1); uistack(p1,'bottom');
    p2 = plot(x_hat_ale, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1); p2.Color(4) = 0.3;
    p3 = plot(x, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2); uistack(p3,'top');
    xlabel('Time Steps (AU)'); ylabel('Amplitude (AU)');
    title(['ALE Evaluation with M=', num2str(M_ale), ' and \Delta=', num2str(D)]);
    [h, ~] = legend([p1, p2, p3], {['$s(n)$'], ['$\hat{x}(n)$'], ['$x(n)$']}, 'Interpreter', 'Latex');
    
    subplot(1,2,2); hold on; hAx = gca; set(hAx,'xminorgrid','on','yminorgrid','on');
    p1 = plot(s, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1); uistack(p1,'bottom');
    p2 = plot(x_hat_anc, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1); p2.Color(4) = 0.3;
    p3 = plot(x, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2); uistack(p3,'top');
    xlabel('Time Steps (AU)'); ylabel('Amplitude (AU)');
    title(['ANC Evaluation with M=', num2str(M_anc)]);
    [g, ~] = legend([p1, p2, p3], {['$s(n)$'], ['$\hat{x}(n)$'], ['$x(n)$']}, 'Interpreter', 'Latex');
end

%Ensemble averaging
mspe = 10*log10(mspe/R);    %MSPE for ALE and ANC
X_ale = X_ale/R;            %Mean ALE filter estimate
X_anc = X_anc/R;            %Mean ANC filter estimate

%PLOT RESULTS
figure; plot(x, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2); hold on; 
plot(X_ale, 'Linewidth', 1); plot(X_anc, 'Color', [0, 0.4470, 0.7410], 'Linewidth', 1); 
grid on; grid minor; legend({'True signal x', 'ALE Estimate', 'ANC Estimate'});
xlabel('Time Steps (AU)'); ylabel('Amplitude (AU)');
title(['Ensemble Averages of ALE and ANC Signal Estimates with M=', num2str(M_anc), ' and \Delta=', num2str(D)]);