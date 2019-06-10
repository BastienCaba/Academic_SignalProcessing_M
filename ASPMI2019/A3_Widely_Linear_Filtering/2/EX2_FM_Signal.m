%%% Adaptive Signal Processing and Machine Intelligence
%%% PART 3: Widely Linear Filtering and Adaptive Spectrum Estimation
clear all; close all; clc;  %Initialise script

%% TASK A: Adaptive AR Model Based Time-Frequency Estimation
%PARAMETERS
N = 1500;       %Number of samples
n = 1:1500;     %Time axis
fs = 2000;      %Sampling frequency (Hz)
var = 0.05;     %Variance of CWGN

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

figure(1); subplot(1,3,1); plot(f, 'Linewidth', 2);
xlabel('Time Index (AU)'); ylabel('Frequency f (Hz)'); grid on; grid minor;
title('Evolution of the time-varying frequency f against time for the non-stationnary FM signal');

%GENERATE FM SIGNAL Y
y = exp(1j*((2*pi)/fs)*phi) + sqrt(var/2)*(randn(1) + 1j*randn(1));
rho = abs(mean((y).^2)/mean(abs(y).^2));  %Circularity coefficient

%BLOCK-BASED AR ESTIMATION
order = [1 2 5];
subplot(1,3,2); hold on;
for p = order
    [A,B] = aryule(y,p);            %Yule-Walker parameters estimation AR(1)
    [h,w] = freqz(sqrt(B),A,N,fs);  %Plot power spectrum density
    plot(w, 10*log10(abs(h).^2), 'Linewidth', 1, 'DisplayName', ['Order ', num2str(p)]);
    xlabel('Frequency (Hz)'); ylabel('Power Spectrum (dB/Hz)'); grid on; grid minor;
    title('Estimated Power Spectrum of FM signal under Yule-Walker for different Orders P');
end
legend show;

subplot(1,3,3); hold on;
for i = 1:3
    z = y(((i-1)*500+1):i*500);
    [A,B] = aryule(z,1);            %Yule-Walker parameters estimation AR(1)
    [h,w] = freqz(sqrt(B),A,N,fs);  %Plot power spectrum density
    plot(w, 10*log10(abs(h).^2), 'Linewidth', 1);
    xlabel('Frequency (Hz)'); ylabel('Power Spectrum (dB/Hz)'); grid on; grid minor;
    title('Estimated Power Spectrum of FM signal under Yule-Walker AR(1) for the 3 Sections of the FM Signal');
end
legend({'f(n) = $100$', 'f(n) = $100 + \frac{n-500}{2}$', 'f(n) = $100 + \Big( \frac{n-1000}{25} \Big)^2$'}, 'Interpreter', 'Latex');

%% TASK B: Adaptive AR Model Based Time-Frequency Estimation
M = 1;                          %AR Model Order
X = complex(zeros(M,N));        %Design Vector X
A = complex(zeros(M,N));        %Filter Weights H
e = complex(zeros(1,N));        %CLMS error
y_hat = complex(zeros(1,N));    %WLMA(1) (estimate)
mu = [0.001 0.01 0.1];          %Learning rate mu

for i = 1:length(mu)
    for n = 2:N
        X(n) = y(n-1);
        [A(n+1), y_hat(n), e(n)] = clms(mu(i), X(n), y(n), A(n));   %CLMS
        [h, w] = freqz(1 , [1; -conj(A(n))], 1024, fs);             %Compute power spectrum
        H(:, n) = abs(h).^2;                                        %Store spectrum power in a matrix
    end
    
    %Frequency estimate
    [~, index] = max(H);
    f_hat = w(index);
    
    %Remove outliers in the matrix H
    medianH = 50*median(median(H));
    H(H>medianH) = medianH;
    
    %Plot time-frequency diagram and frequency estimate
    figure(3); subplot(1,3,i); surf(1:N, w, H, 'Linestyle', 'None'); view(2);
    c = colorbar; xlabel('Time Index (AU)', 'Fontsize', 12); ylabel('Frequency (Hz)', 'Fontsize', 14);
    c.Label.String = 'Power Spectral Density (dB/Hz)';
    title('Spectrogram for the CLMS-AR Model applied to the FM Signal');
    figure(4); subplot(1,3,i); plot(1:N, f_hat, 'Linewidth', 1); hold on; grid on; grid minor;
    plot(1:N, f, 'Linewidth', 1); xlabel('Time Index (AU)', 'Fontsize', 12); ylabel('Frequency (Hz)', 'Fontsize', 14);
    title('Estimate of the frequency f_0 against Time for the CLMS-AR Model applied to the FM Signal');
end

%Circularity plot
figure; scatter(real(y), imag(y), 'filled'); grid on; grid minor;
xlabel('Real'); ylabel('Imaginary'); legend({['\rho \approx ', num2str(round(rho,3))]});
title('Circularity Plot FM Signal'); axis equal;