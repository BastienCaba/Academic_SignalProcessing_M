%%% Adaptive Signal Processing and Machine Intelligence
%%% PART 3: Widely Linear Filtering and Adaptive Spectrum Estimation
clear all; close all; clc;  %Initialise script

%% TASK E: Implement DFT-CLMS algorithm on FM signal
%PARAMETERS
N = 1500;       %Number of samples
t = 1:N;        %Time axis
fs = 1000;      %Sampling frequency (Hz)
var = 0.05;     %Variance of CWGN
mu = 1;         %Learning rate for DFT-CLMS
K = 2048;       %Number of DFT points
gamma = [0.001, 0.01, 0.5];     %Leakage coefficient

%Initialise variables
X = zeros(K,N);     %DFT-CLMS input vector
w = zeros(K,N);     %Weight vector
yh = zeros(1,N);    %Filter output
e = zeros(1,N);     %Filter error
f_axis = linspace(1,fs,K);

%GENERATE PHASE PHI
f = zeros(1,N);
for i = 1:N
    if i <= 500
        f(i) = 100;
    elseif i <= 1000
        f(i) = 100 + (i-500)/2;
    else
        f(i) = 100 + ((i-1000)/25)^2;
    end
end
phi = cumsum(f);
figure(1); subplot(1,length(gamma)+1,1); plot(f, 'Linewidth', 2);
xlabel('Time Index (AU)', 'Fontsize', 12); ylabel('Frequency f (Hz)', 'Fontsize', 14); grid on; grid minor;
title('Evolution of the time-varying frequency f against time for the non-stationnary FM signal');

%GENERATE FM SIGNAL Y
y = exp(1j*((2*pi)/fs)*phi) + sqrt(var/2)*(randn(1) + 1j*randn(1));

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
    medianW = 1000*median(median(W));
    W(W>medianW) = medianW;
    
    %Plot time-frequency diagram and frequency estimate
    subplot(1,length(gamma)+1,i+1); surf(1:N, f_axis, W, 'Linestyle', 'None'); view(2); ylim([50 500]);
    c = colorbar; xlabel('Time Index (AU)', 'Fontsize', 12); ylabel('Frequency (Hz)', 'Fontsize', 14);
    c.Label.String = 'Power Spectral Density (dB/Hz)';
    title(['Spectrogram for the Leaky DFT-CLMS Model applied to the FM Signal (with \gamma=', num2str(gamma(i)),')']);
end