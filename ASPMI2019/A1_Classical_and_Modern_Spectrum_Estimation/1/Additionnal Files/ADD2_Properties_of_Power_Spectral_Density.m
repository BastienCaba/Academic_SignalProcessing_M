close all; clear all; clc;

%% 1.1: Discrete Fourier Transform (DFT)
%Sine wave 20Hz
f_sig = 20;                     %Frequency of the sine (Hz)
fs = 1000;                      %Sampling frequency for the sine (Hz)
N = 100;                        %Number of time samples for the sine
T = 1000*(N/fs);                %Signal duration (ms)
t = (0:1000/fs:T-1);            %Time axis for the sine (ms)
x = sin((2*pi*f_sig/1000)*t);   %Sine signal at 20Hz
figure; stem(t,x);              %Plot sine signal
title([num2str(f_sig), 'Hz sine wave sampled at ', num2str(fs), 'Hz']);
xlabel('Time (ms)'); ylabel('Amplitude (AU)');
grid on; grid minor;

%DFT for different K values
figure; K = [100, 1000, 10000];             %Number of DFT samples
for i = 1:length(K)
    s(i) = subplot(length(K),1,i);
    X = abs(fftshift(fft(x,K(i))));         %DFT magnitude on [-pi:+pi] interval
    freqAxis = fs*(-K(i)/2:K(i)/2-1)/K(i);  %Frequency axis (Hz)
    stem(s(i), freqAxis, X);                %Plot DFT
    title([num2str(K(i)), '-point DFT magnitude spectrum of a ', num2str(f_sig), 'Hz sine wave sampled at ', num2str(fs), 'Hz']);
    xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
    grid on; grid minor;
end

%Sine wave 24Hz
figure;
f_sig = 24;                     %Frequency of the sine (Hz)
x = sin((2*pi*f_sig/1000)*t);   %Sine signal at 24Hz
s(1) = subplot(2,1,1);          %Subplot indexing
stem(s(1), t,x);                %Plot sine signal
title([num2str(f_sig), 'Hz sine wave sampled at ', num2str(fs), 'Hz']);
xlabel('Time (ms)'); ylabel('Amplitude (AU)');
grid on; grid minor;

index = 1;                                          %Select K=100
X = abs(fftshift(fft(x,K(index))));                 %DFT magnitude on [-pi:+pi] interval
freqAxis = fs*(-K(index)/2:K(index)/2-1)/K(index);  %Frequency axis (Hz)
s(2) = subplot(2,1,2);                              %Subplot indexing
stem(s(2), freqAxis, X);                            %Plot DFT
title([num2str(K(index)), '-point DFT magnitude spectrum of a ', num2str(f_sig), 'Hz sine wave sampled at ', num2str(fs), 'Hz']);
xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
grid on; grid minor; clear index;

%Solving the incoherent sampling problem
%Method 1: Increase the number of DFT points K
figure;
index = 2;                                          %Select K=1000
X = abs(fftshift(fft(x,K(index))));                 %DFT magnitude on [-pi:+pi] interval
freqAxis = fs*(-K(index)/2:K(index)/2-1)/K(index);  %Frequency axis (Hz)
s(1) = subplot(3,1,1);                              %Subplot indexing
stem(s(1), freqAxis, X);                            %Plot DFT
title([num2str(K(index)), '-point DFT magnitude spectrum of a ', num2str(f_sig), 'Hz sine wave sampled at ', num2str(fs), 'Hz']);
xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
grid on; grid minor; clear index;

%Method 2: Reducing the sampling frequency
fs = 200;                       %Sampling frequency for the sine (Hz)
T = 1000*(N/fs);                %Signal duration (ms)
t = (0:1000/fs:T-1);            %Time axis for the sine (ms)
x = sin((2*pi*f_sig/1000)*t);   %Sine signal at 20Hz

s(2) = subplot(3,1,2);          %Subplot indexing
stem(t,x);                      %Plot newly sampled sine signal
title([num2str(f_sig), 'Hz sine wave sampled at ', num2str(fs), 'Hz']);
xlabel('Time (ms)'); ylabel('Amplitude (AU)');
grid on; grid minor;

index = 1;                                          %Select K=100
X = abs(fftshift(fft(x,K(index))));                 %DFT magnitude on [-pi:+pi] interval
freqAxis = fs*(-K(index)/2:K(index)/2-1)/K(index);  %Frequency axis (Hz)
s(3) = subplot(3,1,3);                              %Subplot indexing
stem(s(3), freqAxis, X);                            %Plot sine signal
title([num2str(K(index)), '-point DFT magnitude spectrum of a ', num2str(f_sig), 'Hz sine wave sampled at ', num2str(fs), 'Hz']);
xlabel('Time (ms)'); ylabel('Amplitude (AU)');
grid on; grid minor;

%% 1.2.1 Approximation in the Definition of PSD
figure;
fs = 1000;                                          %Sampling frequency for the sine (Hz)
N = 1000*fs;                                        %Time-domain signal length
T = 1000*(N/fs);                                    %Signal duration (ms)
t = (0:1000/fs:T-1);                                %Time axis for the sine (ms)
f_sig = 200;                                        %Frequency of the sine (Hz)

%Generate pink noise sequence (non-decaying ACF)
decay = 10^5;                                       %Exponential rate of decay of magnitude spectrum
M = floor(N/2)-1;                                   %Length of spectrum in each direction (two-sided)
spctr = rand(1, M) .* exp(-(1:M)/decay);            %Amplitude spectrum
spctr = [spctr 0 0 spctr(:, end:-1:1)];             %Amplitude spectrum (extended)
Xtemp = spctr.*exp(2*pi*rand(1, length(spctr))*i);  %Fourier spectrum
x = real(ifft(Xtemp));                              %Pink noise (nondecaying ACF)
x = zscore(x);                                      %Mean = 0, Standard Deviation = 1
%x = sin((2*pi*f_sig/1000)*t);

%Gaussian white noise WGN (decaying ACF)
y = randn(1, N);                                    %Gaussian noise (decaying ACF)

%Deterministic ramp (non-decaying ACF)
z = sin((2*pi*f_sig/1000)*t) + 5*randn(1, N);

index = 1;                                          %Select K=100
X = abs(fftshift(fft(x,K(index))));                 %DFT magnitude on [-pi:+pi] interval for pink noise
Y = abs(fftshift(fft(y,K(index))));                 %DFT magnitude on [-pi:+pi] interval for white Gaussian noise
Z = abs(fftshift(fft(z,K(index))));                 %DFT magnitude on [-pi:+pi] interval for ramp with WGN
freqAxis = fs*(-K(index)/2:K(index)/2-1)/K(index);  %Frequency axis (Hz)

% Computing the PSD with Definition 7
P7x = fftshift(fft(autocorr(x, N-1), K(index)));     %Definition (7) of PSD: DTFT of ACF (for pink noise)
P7y = fftshift(fft(autocorr(y, N-1), K(index)));     %Definition (7) of PSD: DTFT of ACF (for white Gaussian noise)
P7z = fftshift(fft(autocorr(z, N-1), K(index)));     %Definition (7) of PSD: DTFT of ACF (for ramp with added WGN)
s(1) = subplot(2,3,1);                               %Subplot indexing
stem(s(1), freqAxis, P7x);                           %Plot P7 (for pink noise)
title([num2str(K(index)), '-point DFT spectrum of the ACF of a pink noise signal sampled at ', num2str(fs), 'Hz (Definition 7 of PSD)']);
xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
grid on; grid minor;
s(2) = subplot(2,3,2);                              %Subplot indexing
stem(s(2), freqAxis, P7y);                          %Plot P7 (for WGN)
title([num2str(K(index)), '-point DFT spectrum of the ACF of a Gaussian noise signal sampled at ', num2str(fs), 'Hz (Definition 7 of PSD)']);
xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
grid on; grid minor;
s(3) = subplot(2,3,3);                              %Subplot indexing
stem(s(3), freqAxis, P7z);                          %Plot P7 (for ramp with added WGN)
title([num2str(K(index)), '-point DFT spectrum of the ACF of a ramp with added WGN signal sampled at ', num2str(fs), 'Hz (Definition 7 of PSD)']);
xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
grid on; grid minor;

% Computing the PSD with Definition 9
P9x = (1/N).*X.^2;                                  %Definition (9) of PSD: Power of the DTFT of the signal (for pink noise)
P9y = (1/N).*Y.^2;                                  %Definition (9) of PSD: Power of the DTFT of the signal (for WGN)
P9z = (1/N).*Z.^2;                                  %Definition (9) of PSD: Power of the DTFT of the signal (for ramp + WGN)
s(4) = subplot(2,3,4);                              %Subplot indexing
stem(s(4), freqAxis, P9x);                          %Plot P9 (for pink noise)
title([num2str(K(index)), '-point DFT spectrum power of a pink noise signal sampled at ', num2str(fs), 'Hz (Definition 9 of PSD)']);
xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
grid on; grid minor;
s(5) = subplot(2,3,5);                              %Subplot indexing
stem(s(5), freqAxis, P9y);                          %Plot P9 (for WGN)
title([num2str(K(index)), '-point DFT spectrum power of a Gaussian noise signal sampled at ', num2str(fs), 'Hz (Definition 9 of PSD)']);
xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
grid on; grid minor;
s(6) = subplot(2,3,6);                              %Subplot indexing
stem(s(6), freqAxis, P9z);                          %Plot P9 (for ramp with added WGN)
title([num2str(K(index)), '-point DFT spectrum power of a ramp with added WGN signal sampled at ', num2str(fs), 'Hz (Definition 9 of PSD)']);
xlabel('Frequency (Hz)'); ylabel('Magnitude (AU)');
grid on; grid minor;

%% 1.2.2 Using DTFT to determine PSD from ACF
M = [10, 128];          %Variable M
L = 256;                %Length of signals x and z
x = zeros(1,L);         %Initialize signal x
z = zeros(1,L);         %Initialize signal z

figure;
%Signal X
for i = 1:length(M)
    %Forward Pass
    for k = 1:L                                     %Forward iteration through x
        if(abs(k)<=M(i))                            %Function condition
            x(k) = (M(i)-(k-1))/M(i);               %Function expression
        else
            x(k) = 0;                               %Zero-padding
        end
    end
    
    %Backward Pass
    for m = L:(-1):1                                %Backward iteration through x
        k = m - (L+1);                              %Reconstructing variable k
        if(abs(k)<=M(i))                            %Function condition
            x(m) = x(m) + (M(i)-abs(k))/M(i);       %Function expression
        else
            x(m) =  x(m) + 0;                       %Zero-padding
        end
    end
    s(i) = subplot(2*length(M),1,i);                %Subplot indexing
    stem(s(i), 1:L, x);                             %Plot signal x
    title(['Signal x following equation (12) (L: ', num2str(L), ', M: ', num2str(M(i)), ')']);
    xlabel('k'); ylabel('Magnitude (AU)');
    grid on; grid minor;
    
    %FFT{x}
    xf = fft(x);                                        %Compute FFT of x
    s(length(M)+i)=subplot(2*length(M),1,length(M)+i);  %Subplot indexing
    stem((1:L)*2*pi/L,real(xf));                        %Plot FFT{x}
    title(['FFT spectrum of the signal x (L: ', num2str(L), ', M: ', num2str(M(i)), ')']);
    xlabel('Frequency (rads/s)'); ylabel('Magnitude (AU)');
    grid on; grid minor;
end

%Signal Z
figure;
for i = 1:length(M)
    for m = 1:L                                     %Backward iteration through x
        k = m-M(i);                                 %Reconstructing variable k
        if(m<=(2*M(i)))                             %Active section
            z(m) = (M(i)-abs(k))/M(i);              %Function expression
        else
            z(m) =  0;                              %Zero-padding
        end
    end
    
    s(i) = subplot(2*length(M),1,i);                %Subplot indexing
    stem(s(i), 1:L, z);                             %Plot signal x
    title(['Signal z following equation (13) (L: ', num2str(L), ', M: ', num2str(M(i)), ')']);
    xlabel('k'); ylabel('Magnitude (AU)');
    grid on; grid minor;
    
    %FFT{x}
    zf = real(fft(z));                                  %Compute FFT of x
    s(length(M)+i)=subplot(2*length(M),1,length(M)+i);  %Subplot indexing
    stem((1:L)*2*pi/L,zf);                              %Plot FFT{x}
    title(['FFT spectrum of the signal z (L: ', num2str(L), ', M: ', num2str(M(i)), ')']);
    xlabel('Frequency (rads/s)'); ylabel('Magnitude (AU)');
    grid on; grid minor;
end

