%%% ASPMI Coursework 2019: ASSIGNMENT 2
%%% 2.3 Adaptive Noise Cancelation
clear all; close all; clc;  %Initialise script
rng(2);                     %Set seed for random number generation

%% TASK D: Apply ANC to EEG data
%PARAMETERS
R = 50;                 %Number of realisations
M = [1 10 20];          %Filter order
mu = [0.005 0.01 0.1];  %Learning rate

load EEG_Data_Assignment2.mat;
s = zscore(POz);        %Noise-corrupted signal
N = length(s);          %Number of samples
f = 50;                 %Mains frequency (Hz)
t = (0:N-1)/fs;         %Time axis (in s)

win = 4*fs;             %Window length (# samples)
ovp = 50;               %Overlap (%)
K = 12000;              %Number of DFT samples
nov = floor((ovp/100)*win); %Number of samples of overlap

%Spectrogram of the raw POz data
spectrogram(s, hamming(win), nov, K, fs, 'yaxis'); ylim([0 60]);
title('Spectrogram of the raw EEG signal POz');

figure; index = 0;
for j = 1:length(M)         %Iterate over filter orders
    for k = 1:length(mu)    %Iterate over learning rates
        X_anc = zeros(1,N); %ANC filter input
        index = index + 1;  %Plot index
        for i = 1:R         %Iterate over realisations
            epsilon = sin(2*pi*f*t) + random('Normal',0,0.01,1,N);%Secondary noise
            [x_hat_anc, out_anc] = fANC(s, epsilon, mu(k), M(j));   %ANC Filter
            X_anc = X_anc + x_hat_anc;                              %Accumulate ANC output
        end
        X_anc = X_anc(M(j):end)/R;                                  %Mean ANC filter estimate
        subplot(length(M), length(mu), index);
        spectrogram(X_anc, hamming(win), nov, K, fs, 'yaxis'); ylim([0 60]);
        title(['Spectrogram of ANC applied to EEG signal POz (M=', num2str(M(j)), ', \mu=', num2str(mu(k)), ')'])
    end
end

%% TASK D: SHOW ANC OUTPUT FOR OPTIMAL PARAMETERS
M = 10;     %Optimal filter order
mu = 0.01;  %Optimal learning rate
X_anc = zeros(1,N); %ANC filter input

for i = 1:R                                                 %Iterate over realisations
    epsilon = sin(2*pi*f*t) + random('Normal',0,0.01,1,N);  %Secondary noise
    [x_hat_anc, out_anc] = fANC(s, epsilon, mu, M);         %ANC Filter
    X_anc = X_anc + x_hat_anc;                              %Accumulate ANC output
end
X_anc = X_anc(M:end)/R;                                     %Mean ANC filter estimate
difference = s(M:end)' - X_anc;                             %Difference signal: estimate of the noise
periodogram(difference, [], K, fs);                         %Power spectrum of error signal