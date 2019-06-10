%%% ASPMI Coursework 2019: ASSIGNMENT 1
%%% 1.2 Periodogram-based Methods Applied to Real–World Data
clear all; close all; clc;  %Script initialization

%% TASK A: Apply periodogram-based methods to sunspot data
%1. LOAD sunspot data
load sunspot.dat;
x = sunspot(:,2);     %Wolf (relative sunspot) number data sampled yearly
t = sunspot(:,1);     %Time axis from year 1700 to 1987

%2. PREPROCESS sunspot data
x = x/std(x);                               %Scale by the standard deviation
x_n0 = x(find(x ~= 0));                     %Non-zero elements of sunspot data (ignore x=0 elements)
t_n0 = t(find(x ~= 0));                 	%Time axis for nonzero sunspot data

%3. PROCESS sunspot data
x_c = x - mean(x);                      	%Centered sunspot data (signal mean removed)
x_lin_det = detrend(x);                  	%Detrended sunspot data (linear signal trend removed)
x_c_log = log10(x_n0) - mean(log10(x_n0));  %Centered logarithm scaled sunspot data (logarithm applied, mean removed)

signal = {x, x_c, x_lin_det, x_c_log};   	%Processed data signals
time = {t, t, t, t_n0};                    	%Time axes for data signals
name = {'Original', 'Centered', 'Linearly Detrended', 'Centered Logarithmically Scaled'};

%4. COMPUTE PERIODOGRAM for sunspot data
%Windows definitions
win_name = {'Rectangular Window', 'Hanning Window', 'Hamming Window'};
num_windows = length(win_name);

figure(1); hold on;
for j = 1:num_windows                                   %Iterate over window shapes
    for i = 1:length(signal)                            %Iterate over processed data signals
        sunspot_signal = signal{i};                     %Select data signal
        sunspot_timeax = time{i};                       %Select time axis
        subplot(2,4,i);                                 %Subplot indexing
        plot(sunspot_timeax, sunspot_signal);           %PLOT signal
        title([name{i}, ' Sunspot Data (scaled by standard deviation)']);
        xlabel('Year'); ylabel('Wolf number (AU)'); grid on; grid minor; xlim([1700, 1987]);
        
        %Peridogram-based spectral estimation
        window_periodogram = {[], hanning(length(sunspot_signal)), hamming(length(sunspot_signal))};
        subplot(2,length(window_periodogram),j+3);      %Subplot indexing
        [sunspot_Pxx, w] = periodogram(sunspot_signal, window_periodogram{j}, [], 1);
        plot(w, 10*log10(sunspot_Pxx), 'LineWidth',1);  hold on; ylim([-100 100]);
        title(['Periodogram of Relative Sunspot Number Data (', win_name{j}, ')']);
        xlabel('Frequency (Cycles/Year)'); ylabel('Power Spectrum (dB/(Cycles/Year))'); grid on; grid minor;
        legend(name);
    end
end

%5. PLOT RESULTS
figure(2); subplot(1,2,1); plot(t, x, 'LineWidth',1); title('Original and Logarithmically Scaled Sunspot Data (pre-scaled by standard deviation)');
xlabel('Time (Years)'); ylabel('Wolf number (AU)'); grid on; grid minor; xlim([1700, 1987]);
hold on; plot(t_n0, x_c_log, 'LineWidth',1); legend({'Original Data', 'Centered Logarithmically Scaled Data'});

for i = 1:length(signal)                            %Iterate over processed data signals
    sunspot_signal = signal{i};                     %Select data signal
    sunspot_timeax = time{i};                       %Select time axis
    
    %Hamming-windowed periodogram for spectral estimation
    window_periodogram = {[], hanning(length(sunspot_signal)), hamming(length(sunspot_signal))};
    [sunspot_Pxx, w] = periodogram(sunspot_signal, window_periodogram{j}, [], 1);
    subplot(1,2,2); plot(w, 10*log10(sunspot_Pxx), 'LineWidth',1);  hold on; ylim([-100 100]);
    title('Periodogram of Relative Sunspot Number Data (Hamming Window)');
    xlabel('Frequency (Cycles/Year)'); ylabel('Power Spectrum (dB/(Cycles/Year))'); grid on; grid minor;
end
legend(name);

%% TASK B: Apply periodogram-based methods to EEG data
%1. LOAD EEG data
load EEG_Data_Assignment1.mat;                          %EEG recorded from an electrode located at the posterior/occipital (POz) region
var = zeros(1,4);                                       %Variance of periodogram trace

%2. Compute STANDARD PERIODOGRAM
[EEG_Pxx, w] = periodogram(POz, [], 5*fs, fs);          %Compute the PERIODOGRAM with 5 DFT samples/Hz
figure(3); plot(w, 10*log10(EEG_Pxx), 'LineWidth',1);   %Plot the STANDARD PERIODOGRAM
title('Power Spectrum of EEG Data obtained using the Periodogram Estimator');
xlabel('Frequency (Hz)'); ylabel('Power Spectrum (dB/Hz)'); grid on; grid minor; hold on; xlim([0, 60]);
var(1) = std(10*log10(EEG_Pxx))^2;

%3. Compute AVERAGED PERIODOGRAM (Welch's Method)
win_len = [1, 5, 10].*fs;                               %Window lengths to be tried (1s, 5s, 10s)
for i = 1:length(win_len)                               %Iterate over window lengths
    [EEG_Pxx_Av, w] = pwelch(POz, rectwin(win_len(i)), 0, 5*fs, fs);
    plot(w, 10*log10(EEG_Pxx_Av), 'LineWidth',1); 
    var(i+1) = std(10*log10(EEG_Pxx_Av))^2;
end
legend({'Standard Periodogram', 'Averaged Periodogram (Window duration: 1s, no overlap)', 'Averaged Periodogram (Window duration: 5s, no overlap)', 'Averaged Periodogram (Window duration: 10s, no overlap)'});