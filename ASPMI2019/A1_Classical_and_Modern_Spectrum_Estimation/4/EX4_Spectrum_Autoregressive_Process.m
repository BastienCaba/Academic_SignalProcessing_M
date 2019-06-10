%%% ASPMI Coursework 2019: ASSIGNMENT 1
%%% 1.4 Spectrum of Autoregressive Processes
clear all; close all; clc;  %Script initialization

%% TASK: Explore AR processes
%PARAMETERS
order = 4;                      %Order of AR process
x(1:order) = randn(1,order);    %Initial condition (random)
N = 10000;                       %Number of samples in trace generated
w = linspace(-pi, pi, N/2);     %Frequency axis for plotting the PSD

%SIGNAL GENERATION
for n = order:N-1
    x(n+1) = 2.76*x(n)-3.81*x(n-1)+2.65*x(n-2)-0.92*x(n-3)+randn;
end
x = x(500:end);                 %Crop out the TRANSIENT RESPONSE

%TRUE PSD computation
for k = 1:order
    s_v(k,:) = exp(-1*1i*k*w);  %Generate a set of BASIS EXPONENTIAL vectors (steering vectors)
end
A_true = [2.76; -3.81; 2.65; -0.92]; noise_true = 1;            %True parameters a and sigma
H_true2_sided = noise_true./(abs(1-(sum(A_true.*s_v))).^2);     %True PSD (2-sided)
H_true1s = H_true2_sided(1:N/2);                                %True PSD (1-sided)

%AR PARAMETERS ESTIMATION for different MODEL ORDERS
for p = 2:14                            %Iterate over MODEL ORDERS
    for k = 1:p
        v_exp(k,:) = exp(-1*1i*k*w);    %Generate a set of BASIS EXPONENTIAL vectors (steering vectors)
    end
    [A, noise] = aryule(x,p);           %Yule-Walker algorithm to ESTIMATE PARAMETERS a
    A = A(2:end);                       %Crop out the first coefficient (1)
    
    %PLOT TRUE PSD
    figure; plot(w, 20*log10(H_true1s)); title('True and Estimated PSD of an Autoregressive Process AR(4)'); hold on;
    xlabel('Frequency (rad/s)'); ylabel('Power/frequency (dB/(rad/s))'); xlim([0, pi]); grid on; grid minor;
    
    %Compute FILTER TRANSFER FUNCTION
    H2sided = noise./(abs(1+(sum(A'.*v_exp))).^2);  %Two-sided power spectrum estimate
    H1sided{p} = H2sided(1:N/2);                    %Single-sided power spectrum estimate
    plot(w, 20*log10(H1sided{p}), 'Linewidth', 1);  %Plot power spectrum (in dB)
    legend({'True PSD',['PSD Estimate with AR(', num2str(p), ')']});
    
    %Compute ERROR
    error(p-1) = mean(20*log10(H1sided{p}) - 20*log10(H_true1s));
    clear v_exp; clear A; clear noise;
end

%PLOT ERROR
figure; plot(2:14, error, 'Linewidth', 2); title('Evolution of the Mean Error between the True and Estimated Power Spectrum for the signal of AR(4)');
xlabel('Model order p'); ylabel('Mean Error (dB)'); xlim([1, 15]); grid on; grid minor; xlim([2 14]);