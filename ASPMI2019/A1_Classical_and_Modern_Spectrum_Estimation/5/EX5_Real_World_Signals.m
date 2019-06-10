%%% ASPMI Coursework 2019: ASSIGNMENT 1
%%% 1.5 Real World Signals: Respiratory Sinus Arrhythmia from RR-Intervals
clear all; close all; clc;  %Script initialisation

%% LOAD and CONVERT ECG DATA (only run once)
% load('ECG.mat');
% Trial_1 = data(find(time==64):find(time==237.1));
% Trial_2 = data(find(time==241):find(time==489.4));
% Trial_3 = data(find(time==491):find(time==737.5));
% [RRI_1, RRIf_1] = ECG_to_RRI(Trial_1,fs);
% [RRI_2, RRIf_2] = ECG_to_RRI(Trial_2,fs);
% [RRI_3, RRIf_3] = ECG_to_RRI(Trial_3,fs);
% save('RRI_1.mat','RRI_1','RRIf_1');
% save('RRI_2.mat','RRI_2','RRIf_2');
% save('RRI_3.mat','RRI_3','RRIf_3');

%% TASK: Spectrum estimation of RRI data
load('RRI_1.mat');  %Load RRI data and sampling frequency for TRIAL 1
load('RRI_2.mat');  %Load RRI data and sampling frequency for TRIAL 2
load('RRI_3.mat');  %load RRI data and sampling frequency for TRIAL 3

%PREPROCESSING
RRI_1 = detrend(zscore(RRI_1));     %Data linearly DETRENDED and STANDARDIZED
RRI_2 = detrend(zscore(RRI_2));     %Data linearly DETRENDED and STANDARDIZED
RRI_3 = detrend(zscore(RRI_3));     %Data linearly DETRENDED and STANDARDIZED

%COMPILE SIGNALS
RRI_sig = {RRI_1, RRI_2, RRI_3};
RRIf = {RRIf_1, RRIf_2, RRIf_3};
TrialName = {'Normal Breathing', 'Fast Breathing: 25 breaths/min', 'Slow Breathing: 7.5 breaths/min'};

%SPECTRAL ESTIMATION
figure;
for i = 1:length(RRI_sig)
    %Classical Spectrum Estimation: PERIODOGRAM
    subplot(1, length(RRI_sig), i); hold on;    %Figure initialisation
    data = RRI_sig{i};                          %Select RRI data for TRIAL
    fs = RRIf{i};                               %Select sampling frequency for TRIAL
    trial = TrialName{i};                       %Select TRIAL name
    win_len = [150 50]*fs;                      %Window length vector for averaged periodogram
    [RRI_pxx, f_axis] = periodogram(data, hamming(length(data)), [], 60*fs);        %Periodogram
    p1 = plot(f_axis, 10*log10(RRI_pxx), 'DisplayName', 'Standard Periodogram');    %Plot results
    
    %Classical Spectrum Estimation: AVERAGED PERIODOGRAM
    for j = 1:length(win_len)
        [RRI_pxx_av, f_axis_av] = pwelch(data, hamming(win_len(j)), [], [], 60*fs); %Averaged periodogram
        p2(j) = plot(f_axis_av, 10*log10(RRI_pxx_av), 'DisplayName', ['Averaged Periodogram (Window Length: ', num2str(win_len(j)/fs), 's)']);
    end
    xlabel('Frequency (BPM)'); ylabel('Power/frequency (dB/Hz)'); legend([p1, p2(1), p2(2)]); grid on; grid minor;
    title(['Periodogram-based Power Spectrum Estimates of RRI Data for ', trial, ' (Hamming Window)']);
end

%Modern Spectrum Estimation: AR MODEL
figure; p_order = [1,11,15; 1,7,15; 1,2,15];
for i = 1:length(RRI_sig)                       %Iterate over trials
    subplot(1, length(RRI_sig), i);             %Figure initialisation
    hold on; p = p_order(i,:);                  %AR orders to be tried
    data = RRI_sig{i};                          %Select RRI data for TRIAL
    fs = RRIf{i};                               %Select sampling frequency for TRIAL
    trial = TrialName{i};                       %Select TRIAL name
    [RRI_pxx_av, f_axis_av] = pwelch(data, hamming(50*fs), [], [], 60*fs);
    p1 = plot(f_axis_av, 10*log10(RRI_pxx_av), 'DisplayName', 'Averaged Periodogram (Window Length: 50s)');
    for j = 1:length(p)                         %Iterate over model orders
        [A, noise] = aryule(data,p(j));         %Yule-Walker algorithm to ESTIMATE PARAMETERS a
        AR_pxx = abs(freqz(noise, A, f_axis_av, 60*fs)).^2; %PSD ESTIMATE
        p2(j) = plot(f_axis_av, 10*log10(AR_pxx), 'DisplayName', ['AR(', num2str(p(j)), ') Model']);
    end
    legend([p1, p2(1:end)], 'Location', 'Southwest'); grid on; grid minor; xlabel('Frequency (BPM)'); ylabel('Power/frequency (dB/Hz)');
    title(['Comparison of Periodogram-based Spectrum Estimation and AR Modeling on RRI Data for ', trial]);
end