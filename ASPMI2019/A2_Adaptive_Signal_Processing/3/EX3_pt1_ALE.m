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

D = 1:25;           %Delay delta
M = 1:20;           %Filter order
mspe = zeros(length(D), length(M));

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
    
    for j = 1:length(D)
        for k = 1:length(M)
        [e, x_hat] = fALE(s, D(j), M(k), 0.01);
        mspe(j,k) = mspe(j,k) + mean((x(M(k):end)-x_hat(M(k):end)).^2);
        end
    end
end
mspe = 10*log10(mspe/R);

%PLOT RESULTS
figure; subplot(1,3,1); grid on; grid minor;
surf(M(5:end), D, mspe(:,5:end), 'Facecolor', 'interp'); xlabel('Filter Order M'); ylabel('Delay \Delta');
zlabel('MSPE (dB)'); title('Surface Plot of MSPE against Filter Order M and Delay \Delta for ALE');
subplot(1,3,2); plot(mspe(:,5)); xlabel('Delay \Delta'); ylabel('MSPE (dB)');
grid on; grid minor; title('MSPE against Delay \Delta for ALE (M = 5)');
subplot(1,3,3); plot(mspe(3,:)); xlabel('Filter Order M'); ylabel('MSPE (dB)');
grid on; grid minor; title('MSPE against Filter Order M for ALE (\Delta = \Delta_{min})');