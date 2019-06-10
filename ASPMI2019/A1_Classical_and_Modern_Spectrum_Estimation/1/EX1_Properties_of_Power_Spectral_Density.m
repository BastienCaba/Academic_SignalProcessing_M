%%% ASPMI Coursework 2019: ASSIGNMENT 1
%%% 1.1 Properties of Power Spectral Density (PSD)
clear all; close all; clc;  %Script initialization

%% TASK: Demonstrate the equivalence of two different definitions of PSD
% Generate test signal
fs = 1000;                                      %Signal sampling frequency (Hz)
N = fs;                                         %Signal length (1s)
wgn = zscore(randn(1,N));                       %White Gaussian Noise test signal

% AUTOCORRELATION ESTIMATION (biased and unbiased)
[rxx_unb, k_unb] = autocorr_unbiased(wgn);      %Unbiased autocorrelation estimator
rxx_unb = [fliplr(rxx_unb(2:end)) rxx_unb];     %Two-sided expansion
[rxx_bias, k_bias] = autocorr_biased(wgn);      %Biased autocorrelation estimator
rxx_bias = [fliplr(rxx_bias(2:end)) rxx_bias];  %Two-sided expansion

% CORRELOGRAM
pxx_b = fftshift(fft(ifftshift(rxx_bias), length(rxx_bias)));  	%Correlogram from biased autocorrelation
pxx_u = fftshift(fft(ifftshift(rxx_unb, length(rxx_bias))));   	%Correlogram from unbiased autocorrelation
pxx_b = pxx_b(1000:end);                        %ONE-SIDED for comparison with periodogram
pxx_u = pxx_u(1000:end);                        %ONE-SIDED for comparison with periodogram

% PERIODOGRAM
[pxx_periodogram, f] = periodogram(wgn, rectwin(length(wgn)), 2*length(pxx_u)-1, fs);

% PLOT RESULTS: non-equivalence with unbiased correlogram
subplot(1,2,1); plot(f, pxx_u); hold on;
plot(f, length(pxx_periodogram)/2*pxx_periodogram);                         %Plot normalised periodogram
error_u = abs(norm(pxx_u - (length(pxx_periodogram)/2)*pxx_periodogram'));  %Error metric E
title(['Simulation Case where Definitions (7) and (9) are NOT Equivalent (E=', num2str(error_u), ')']);
grid on; grid minor; xlabel('Frequency (Hz)'); ylabel('Power/frequency (AU/Hz)');
legend({'Correlogram', 'Periodogram'}); xlim([0 fs/2]);

% PLOT RESULTS: equivalence with biased correlogram
subplot(1,2,2); plot(f, pxx_b); hold on;
plot(f, length(pxx_periodogram)/2*pxx_periodogram);                         %Plot normalised periodogram
error_b = abs(norm(pxx_b - (length(pxx_periodogram)/2)*pxx_periodogram'));  %Error metric E
title(['Simulation where Definitions (7) and (9) are Equivalent (E=', num2str(error_b), ')']);
grid on; grid minor; xlabel('Frequency (Hz)'); ylabel('Power/frequency (AU/Hz)');
legend({'Correlogram', 'Periodogram'}); xlim([0 fs/2]);