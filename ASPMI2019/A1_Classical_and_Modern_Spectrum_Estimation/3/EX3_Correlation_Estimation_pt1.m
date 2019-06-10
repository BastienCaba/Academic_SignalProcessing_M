%%% ASPMI Coursework 2019: ASSIGNMENT 1
%%% 1.3 Correlation Estimation (pt. 1)
clear all; close all; clc;  %Script initialization

%% TASK: Explore biased and unbiased correlation estimation
%PARAMETERS
fs = 1000;                                  %Signal sampling frequency (Hz)
duration = 1;                               %Signal duration (s)
N = duration*fs;                            %Signal length
t = (0:1000/fs:(1000*duration)-1);          %Time axis for the sine (ms)

%NOISY SINE
f_sig = 200;                                                        %Frequency of the sine (Hz)
s_det = zscore(sin((2*pi*f_sig/1000)*t));                           %Sine wave (determinsitic)
alpha = 4;                                                          %Amplitude ratio of noise to sine
noisy_sine = zscore(sin((2*pi*f_sig/1000)*t) + alpha*randn(1,N));   %Sine wave with WGN (White Gaussian Noise)

%WHIT GAUSSIAN NOISE
wgn = zscore(randn(1,N));                                           %White Gaussian Noise signal

%PINK NOISE (filtered WGN)
decay = 10^2;                                                       %Exponential rate of decay of magnitude spectrum
M = floor(N/2)-1;                                                   %Length of spectrum in each direction (two-sided)
spctr = rand(1, M) .* exp(-(1:M)/decay);                            %Amplitude spectrum
spctr = [spctr 0 0 spctr(:, end:-1:1)];                             %Amplitude spectrum (extended)
Xtemp = spctr.*exp(2*pi*rand(1, length(spctr))*1i);                 %Fourier spectrum
pink_noise = real(ifft(Xtemp));                                     %Pink noise (nondecaying ACF)
pink_noise = zscore(pink_noise);                                    %Mean = 0, Standard Deviation = 1

%COMPILE TEST SIGNALS
x = {ones(1, N), noisy_sine, wgn, pink_noise};
sig_names = {['a Deterministic Sine Wave at ' num2str(f_sig) 'Hz Frequency'] ['a Noisy Sine Wave at ' num2str(f_sig) 'Hz Frequency with White Gaussian Noise (WGN) (SNR = ' num2str(round(10*log10((1/alpha)^2)), 3) ' dB)'], 'White Gaussian Noise (WGN)', 'Pink Noise'};

%AUTOCORRELATION estimators (biased and unbiased)
for i = 1:length(x)
    %SINGLE-SIDED autocorrelation
    figure(i); freqAxis = linspace(-fs/2, fs/2, 2*N-1); %Frequency axis (Hz)
    [rxx_bias, k_bias] = autocorr_biased(x{i});         %Biased autocorrelation estimator
    [rxx_unb, k_unb] = autocorr_unbiased(x{i});         %Unbiased autocorrelation estimator
    
    %DOUBLE-SIDED autocorrelation
    rxx_bias = [fliplr(rxx_bias(2:end)) rxx_bias];      %Mirror autocorrelation (biased)
    k_bias = [-fliplr(k_bias(2:end)) k_bias];           %Mirror lag axis (biased)
    rxx_unb = [fliplr(rxx_unb(2:end)) rxx_unb];         %Mirror autocorrelation (unbiased)
    k_unb = [-fliplr(k_unb(2:end)) k_unb];              %Mirror lag axis (unbiased)
    
    %PLOT RESULTS
    subplot(1,2,1);                                 %Subplot indexing
    plot(k_unb, rxx_unb); hold on;                  %Plot unbiased autocorrelation estimator
    title(['Autocorrelation Estimate for ' sig_names(i)]); grid on; grid minor;
    xlabel('Lag k (ms)'); ylabel('$$\hat{\phi}$$','Interpreter','Latex', 'FontSize', 15);
    plot(k_bias, rxx_bias);                         %Plot biased autocorrelation estimator
    legend({'Unbiased \phi_u', 'Biased \phi_b'});
    
    %COMPUTE correlogram
    pxx_b = fftshift(fft(ifftshift(rxx_bias), length(rxx_bias)));   %Correlogram from biased autocorrelation
    pxx_u = fftshift(fft(ifftshift(rxx_unb, length(rxx_bias))));    %Correlogram from unbiased autocorrelation
    
  	%COMPUTE imaginary to real amplitude
    im_ratio_b = mean(imag(pxx_b));                 %Biased case
    im_ratio_u = mean(imag(pxx_u));                 %Unbiased case
    
    %PLOT RESULTS
    subplot(1,2,2);                                 %Subplot indexing
    plot(freqAxis, real(pxx_u)); hold on;         	%Plot unbiased correlogram
    title(['Correlogram estimate for ', sig_names(i)]); grid on; grid minor;
    xlabel('Frequency (Hz)'); ylabel('Power/frequency (AU/Hz)');
    plot(freqAxis, real(pxx_b));                    %Plot biased correlogram
    legend({['Unbiased (\alpha\approx', num2str(im_ratio_u), ')'], ['Biased (\alpha\approx', num2str(im_ratio_b), ')']});
end

%% TASK: Explore PSD variance
%PARAMETERS
fs = 500;                                   %Signal sampling frequency (in Hz)
duration = 0.5;                             %Signal duration (in s)
N = duration*fs;                            %Signal length (# samples)
K = 10*N;                                   %Number of DFT samples
t = (0:1000/fs:(1000*duration)-1);          %Time axis for the sine (in ms)

R = 100;                                    %Number of realisations
f1 = 100;                                   %Frequency of the first sinusoid
f2 = 150;                                   %Frequency of the second sinusoid
alpha = 1;                                  %Amplitude ratio of noise to each sinusoid
pxx_rel = zeros(R, K/2+1);                  %Matrix of realisations of biased correlogram

%GENERATE REALISATION of random process and COMPUTE CORRELOGRAM
figure;
for i = 1:R                                 %Iterate over realisations
    x = zscore(sin((2*pi*f1/1000)*t) + sin((2*pi*f2/1000)*t) + alpha*randn(1,N));
    [rxx, k] = autocorr_biased(x);          %Biased autocorrelation
    pxx = fft(rxx, K);                      %Correlogram from biased autocorrelation
    
    %SINGLE-SIDED spectrum extraction
    pxx_rel(i,:) = pxx(1:K/2+1);
    freqAxis = fs*(0:(K/2))/K;
    
    subplot(2,1,1); p = plot(freqAxis, pxx_rel(i,:), 'b'); hold on; p.Color(4) = 0.1;
    title(['Biased Correlogram for ', num2str(R), ' Realisations of Noisy Sinusoids with WGN (f_1=', num2str(f1), 'Hz, f_2=', num2str(f2), 'Hz, SNR=', num2str(round(10*log10((1/alpha^2))),3), 'dB)']);
    xlabel('Frequency (Hz)'); ylabel('Power/frequency (AU/Hz)'); grid on; grid minor;
end

%COMPUTING MEAN and STD from all realisations
mu_pxx = mean(pxx_rel);     %Mean correlogram
std_pxx = std(pxx_rel);     %Standard deviation of correlograms

plot(freqAxis, mu_pxx, 'b', 'LineWidth', 1);                    %Plot mean
subplot(2,1,2); plot(freqAxis, std_pxx, 'r', 'LineWidth', 1);   %Plot standard deviation
title(['Standard Deviation of the Correlogram Spectral Estimate for ', num2str(R), ' Realisations of Noisy Sinusoids with WGN (f_1=', num2str(f1), 'Hz, f_2=', num2str(f2), 'Hz, SNR=', num2str(round(10*log10((1/alpha^2))),3), 'dB)']);
xlabel('Frequency (Hz)'); ylabel('Power/frequency (AU/Hz)'); grid on; grid minor;

%% TASK: Explore decibels representation of correlograms
figure;
for i = 1:R                             %Iterate over realisations
    x = zscore(sin((2*pi*f1/1000)*t) + sin((2*pi*f2/1000)*t) + alpha*randn(1,N));
    [rxx, k] = autocorr_biased(x);      %Biased autocorrelation
    pxx = fft(rxx, K);                  %Correlogram with biased autocorrelation
    
    %SINGLE-SIDED spectrum extraction
    pxx_rel(i,:) = 10*log10(pxx(1:K/2+1));
    freqAxis = fs*(0:(K/2))/K;
    
    subplot(2,1,1); p = plot(freqAxis, pxx_rel(i,:), 'b'); hold on; p.Color(4) = 0.1;
    title(['Biased Correlogram for ', num2str(R), ' Realisations of Noisy Sinusoids with WGN (f_1=', num2str(f1), 'Hz, f_2=', num2str(f2), 'Hz, SNR=', num2str(round(10*log10((1/alpha^2))),3), 'dB)']);
    xlabel('Frequency (Hz)'); ylabel('Power/frequency (dB/Hz)'); grid on; grid minor;
end

%COMPUTING MEAN and STD from all realisations
mu_pxx = mean(pxx_rel);     %Mean correlogram
std_pxx = std(pxx_rel);     %Standard deviation of correlograms
plot(freqAxis, mu_pxx, 'b', 'LineWidth', 1);                    %Plot mean
subplot(2,1,2); plot(freqAxis, std_pxx, 'r', 'LineWidth', 1);   %Plot standard deviation
title(['Standard Deviation of the Correlogram Spectral Estimate for ', num2str(R), ' Realisations of Noisy Sinusoids with WGN (f_1=', num2str(f1), 'Hz, f_2=', num2str(f2), 'Hz, SNR=', num2str(round(10*log10((1/alpha^2))),3), 'dB)']);
xlabel('Frequency (Hz)'); ylabel('Power/frequency (dB/Hz)'); grid on; grid minor;