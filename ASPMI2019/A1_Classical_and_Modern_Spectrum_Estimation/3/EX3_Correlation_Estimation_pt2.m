%%% ASPMI Coursework 2019: ASSIGNMENT 1
%%% 1.3 Correlation Estimation (pt. 2)
clear all; close all; clc;  %Script initialization

%% TASK: frequency estimation by MUSIC
%PARAMETERS
fs = 1;                         %Unity signal sampling frequency (in Hz)
N_vec = [30, 50, 100];          %Length of the complex signal (in # samples)
f = [0.3, 0.32; 0.3, 0.31];     %Frequency matrix containing couples f1,f2 as rows
var = 0.2;                      %Variance of the White Gaussian Noise (WGN)

for i = 1:length(f(:,1))        %Iterate over frequency couples
    f1 = f(i, 1);               %Frequency of first complex exponential
    f2 = f(i, 2);               %Frequency of second complex exponential
    for j = 1:length(N_vec)
        %GENERATE COMPLEX SIGNAL
        N = N_vec(j);       %Assign new signal length
        dur = N/fs;         %Resulting signal duration (in s)
        n = 0:1/fs:dur;     %Time axis (in s)
        %Complex noise component (complex White Gaussian Noise)
        noise = var/sqrt(2)*(randn(size(n))+1j*randn(size(n)));
        %Signal x consiting of two complex exponentials with complex WGN
        x = exp(1j*2*pi*f1*n)+exp(1j*2*pi*f2*n)+ noise;
        
        %COMPUTE PERIODOGRAM
        [Pxx, w] = periodogram(x, rectwin(length(n)), 128, fs);            	%Compute the PERIODOGRAM
        subplot(1, 2, i); plot(w, 10*log10(Pxx), 'LineWidth',1); hold on;   %Plot the PERIODOGRAM
        title(['Periodogram of Two Complex Exponentials in complex WGN (f_1=', num2str(f1), 'Hz, f_2=', num2str(f2), 'Hz, \delta_f=', num2str(f2-f1), 'Hz)']);
        xlabel('Frequency (Hz)'); ylabel('Power/frequency (dB/Hz)'); grid on; grid minor; xlim([0, 0.5]);
    end
    legend({['N=', num2str(N_vec(1)), ', N^{-1}=', num2str(round(1/N_vec(1), 3))], ['N=', num2str(N_vec(2)), ', N^{-1}=', num2str(round(1/N_vec(2), 3))], ['N=', num2str(N_vec(3)), ', N^{-1}=', num2str(round(1/N_vec(3), 3))]});
end

%Coursework code
[X,R] = corrmtx(x,50,'mod');    
[S,F] = pmusic(R,2,[ ],1,'corr');
figure; plot(F,S,'LineWidth',2); set(gca,'xlim',[0.25 0.40]);
grid on; grid minor; xlabel('Frequency (Hz)'); ylabel('Pseudospectrum');
title('Pseudospectrum obtained from the MUSIC algorithm');