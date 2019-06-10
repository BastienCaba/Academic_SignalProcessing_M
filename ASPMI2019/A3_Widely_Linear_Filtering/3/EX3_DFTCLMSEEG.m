%%% Adaptive Signal Processing and Machine Intelligence
%%% PART 3: Widely Linear Filtering and Adaptive Spectrum Estimation
clear all; close all; clc;  %Initialise script

%% TASK F: Implement DFT-CLMS algorithm on EEG signal
load EEG_Data_Assignment1.mat; a = 500;
y = zscore(POz(a:a+1200-1));    %Noise-corrupted signal
N = length(y);                  %Number of samples
t = (0:N-1)/fs;                 %Time axis (in s)
mu = 1;                         %Learning rate for DFT-CLMS
K = 4096;                       %Number of DFT points
gamma = [0, 0.01, 0.1];         %Leakage coefficient

%Initialise variables
X = zeros(K,N);     %DFT-CLMS input vector
w = zeros(K,N);     %Weight vector
yh = zeros(1,N);    %Filter output
e = zeros(1,N);     %Filter error
f_axis = linspace(1,fs,K);

%DFT-CLMS
for i = 1:length(gamma)
    for n = 1:N         %Iterate over time
        for k = 0:K-1   %Generate input vector
            X(k+1,n) = (1/K)*exp(1j*2*pi*k*n/K);
        end
        yh(n) = ctranspose(w(:,n))*X(:,n);      %Filter output
        e(n) = y(n) - yh(n);                    %Filter error
        w(:,n+1) = (1-gamma(i)*mu)*w(:,n) + mu*conj(e(n))*X(:,n);
    end
    
    W = abs(w(:,2:end).^2);
    
    %Remove outliers in the matrix W
    medianW = 100*median(median(W));
    W(W>medianW) = medianW;
    
    %Plot time-frequency diagram and frequency estimate
    subplot(1,length(gamma),i); surf(1:N, f_axis, W, 'Linestyle', 'None'); view(2); ylim([0 fs/4]);
    c = colorbar; xlabel('Time Index (AU)', 'Fontsize', 12); ylabel('Frequency (Hz)', 'Fontsize', 14);
    c.Label.String = 'Power Spectral Density (dB/Hz)';
    title(['Spectrogram for the Leaky DFT-CLMS Model applied to the POz Signal (with \gamma=', num2str(gamma(i)),')']);
end