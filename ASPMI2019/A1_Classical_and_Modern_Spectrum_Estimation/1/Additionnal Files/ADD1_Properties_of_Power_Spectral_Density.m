%% 1.1 Properties of Power Spectral Density (PSD)
% TASK: Show through simulations that the definition of PSD (7) is equivalent to (9) under the assumption that r(k) decays rapidly
clear all; close all; clc;

%% PARAMETERS for SIGNAL GENERATION
fs = 100;           %Sampling frequency for both simulations (Hz)
dur = 10;           %Duration of our simple ramp signal x1 (in s)
N = dur*fs;         %Signal length for both simuations (# samples)

%% GENERATE simple signal x1
x1 = zeros(1,N);    %Initialize signal x1
t = (0:N-1)/fs;     %Time axis for x1
f_sig = 20;         %Frequency of sine wave in x1
noise = 5;          %Ratio of noise to signal amplitude

x1 = sin((2*pi*f_sig)*t) + noise*randn(1, N);

%% PLOT simple signal x1
figure(1); plot(t, x1);
title(['Simple Signal x1 to Simulate the Case where PSD Definitions (7) and (9) are Equivalent (N=', num2str(N), ', fs=', num2str(fs), 'Hz)']);
xlabel('Time (s)'); ylabel('Amplitude (AU)'); grid on; grid minor;

%% COMPUTE the ACF of simple signal x1
acf_x1 = autocorr(x1, N-1);         %Unbiased autocorrelation estimate of x1
k = 0:length(x1)-1;                 %Axis of lags k to plot the autocorrelation
figure(2); stem(k, acf_x1);         %Plot the ACF of x1
title(['Autocorrelation (ACF) of Signal x1 to Simulate the Case where PSD Definitions (7) and (9) are Equivalent (N=', num2str(N), ', fs=', num2str(fs), 'Hz)']);
xlabel('Lag k'); ylabel('Autocorrelation (AU)'); grid on; grid minor;

%% COMPUTE PSD based on definition (7)
f = linspace(-fs/2,fs/2-1/fs,N);                            %Define frequency axis (in Hz)
PSD7 = (1/N)*(fftshift(fft(x1)).*conj(fftshift(fft(x1))));  %Definition (7) of PSD
figure(3); subplot(2,1,1); stem(f,PSD7);                   	%Plot PSD (7)
title('Definition (7) of PSD applied to Signal x1 to Simulate the Case where PSD Definitions (7) and (9) are Equivalent');
xlabel('Frequency (Hz)'); ylabel('PSD (Power/frequency) (AU/Hz)'); grid on; grid minor;

%% COMPUTE PSD based on definition (9)
f = linspace(-fs/2,fs/2-1/fs,length(acf_x1));               %Define frequency axis (in Hz)
PSD9 = fftshift(fft(acf_x1));                               %Definition (9) of PSD
figure(3); subplot(2,1,2); stem(f,PSD9);                   	%Plot PSD (9)
title('Definition (9) of PSD applied to Signal x1 to Simulate the Case where PSD Definitions (7) and (9) are Equivalent');
xlabel('Frequency (Hz)'); ylabel('PSD (Power/frequency) (AU/Hz)'); grid on; grid minor;

%% ERROR between (7) and (9)
err = dot(PSD7,PSD9)./(norm(PSD7)*norm(PSD9));