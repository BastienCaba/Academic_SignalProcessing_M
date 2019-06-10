%%% ASPMI Coursework 2019: ASSIGNMENT 2
%%% 2.2 Adaptive Step Size
clear all; close all; clc;  %Initialise script

%PARAMETERS
R = 100;            %Number of realisations
N = 1000;           %Number of samples (signal length)
mu = [0.01, 0.1];   %Learning rate (initial)
a = 0.9;            %MA parameter
sigma = 0.5;        %MA noise parameter

rho = 1e-3;         %Hyperparameter for GASS learning rate update
alpha = 0.8;        %Hyperparameter for Ang & Farhang learning rate update

%Variables Initialisation
e_std = zeros(R, N, length(mu));   %Error evolution matrix, nonadaptive LMS
e_b = zeros(R, N, length(mu));     %Error evolution matrix, adaptive Benveniste
e_af = zeros(R, N, length(mu));    %Error evolution matrix, adaptive Ang & Farhang 
e_mx = zeros(R, N, length(mu));    %Error evolution matrix, adaptive Matthews and Xie

for i = 1:R                         %Iterate over realisations
    %True Signal
    x = zeros(1,N);                 %Initialise true signal x
    
    %Predicted Signals
    x_hat_std = zeros(1,N);         %Initialise predicted signal x_hat (nonadaptive)
    x_hat_b = zeros(1,N);           %Initialise predicted signal x_hat (Benveniste)
    x_hat_af = zeros(1,N);          %Initialise predicted signal x_hat (Ang & Farhang)
    x_hat_mx = zeros(1,N);          %Initialise predicted signal x_hat (Matthews and Xie)
    
    %Error Signals
    e_temp_std = zeros(1,N);        %Initialise error signal e_temp (nonadaptive)
    e_temp_b = zeros(1,N);          %Initialise error signal e_temp (Benveniste)
    e_temp_af = zeros(1,N);         %Initialise error signal e_temp (Ang & Farhang)
    e_temp_mx = zeros(1,N);         %Initialise error signal e_temp (Matthews and Xie)
    
    %Weights
    w_temp_std = randn(length(a), N);   %Initialise weights (AR parameters, nonadaptive)
    w_temp_b = randn(length(a), N);     %Initialise weights (AR parameters, Benveniste)
    w_temp_af = randn(length(a), N);    %Initialise weights (AR parameters, Ang & Farhang)
    w_temp_mx = randn(length(a), N);    %Initialise weights (AR parameters, Matthews and Xie)
    
    %Initial conditions
    for j = 1:length(a)+1
        x(j) = rand(1);                 %Initial condition for the true signal x
        eta(j) = sqrt(sigma).*randn(1); %Initial condition for the noise term eta
        psi_b(j) = rand(1);             %Initial condition for psi_b (Benveniste)
        psi_af(j) = rand(1);            %Initial condition for psi_af (Ang & Farhang)
    end
    
    %% LMS ALGORITHM
    for k = 1:length(mu)
        mu_b(1:3) = mu(k)*ones(1,3);    %Initial learning rate (Benveniste)
        mu_af(1:3) = mu(k)*ones(1,3);   %Initial learnign rate (Ang & Farhang)
        mu_mx(1:3) = mu(k)*ones(1,3);   %Initial learning rate (Matthews and Xie)
        
        for n = length(a)+2:N
            eta(n) = sqrt(sigma)*randn(1);  %Generate noise sample
            x(n) = a*eta(n-1) + eta(n);     %MA(1) Definition of the true signal
            
            %LMS Output
            x_hat_std(n) = w_temp_std(n)*eta(n-1);  %Predicted value (nonadaptive)
            x_hat_b(n) = w_temp_b(n)*eta(n-1);      %Predicted value (Benveniste)
            x_hat_af(n) = w_temp_af(n)*eta(n-1);    %Predicted value (Ang & Farhang)
            x_hat_mx(n) = w_temp_mx(n)*eta(n-1);    %Predicted value (Matthews and Xie)
            
            %LMS Error
            e_temp_std(n) = x(n) - x_hat_std(n);  	%Prediction error (nonadaptive)
            e_temp_b(n) = x(n) - x_hat_b(n);        %Prediction error (Benveniste)
            e_temp_af(n) = x(n) - x_hat_af(n);      %Prediction error (Ang & Farhang)
            e_temp_mx(n) = x(n) - x_hat_mx(n);      %Prediction error (Matthews and Xie)
            
            %Weight Update
            w_temp_std(n+1) = w_temp_std(n) + mu(k).*e_temp_std(n).*eta(n-1);   %Nonadaptive
            w_temp_b(n+1) = w_temp_b(n) + mu_b(n).*e_temp_b(n).*eta(n-1);       %Benveniste
            w_temp_af(n+1) = w_temp_af(n) + mu_af(n).*e_temp_af(n).*eta(n-1);   %Ang & Farhang
            w_temp_mx(n+1) = w_temp_mx(n) + mu_mx(n).*e_temp_mx(n).*eta(n-1);   %Matthews and Xie
            
            %PSI Functions
            psi_b(n) = (1-mu_b(n-1)*(eta(n-2)^2))*psi_b(n-1) + e_temp_b(n-1)*eta(n-2);  %Benveniste
            psi_af(n) = alpha*psi_af(n-1) + e_temp_af(n-1)*eta(n-2);                    %Ang & Farhang
            psi_mx(n) = e_temp_mx(n-1)*eta(n-2);                                        %Matthews & Xie
            
            %Learning Rate Update
            mu_b(n+1) = mu_b(n) + rho*e_temp_b(n)*eta(n-1)*psi_b(n);        %Benveniste
            mu_af(n+1) = mu_af(n) + rho*e_temp_af(n)*eta(n-1)*psi_af(n);    %Ang & Farhang
            mu_mx(n+1) = mu_mx(n) + rho*e_temp_mx(n)*eta(n-1)*psi_mx(n);    %Matthews & Xie
        end
        
        %Error power
        e_std(i,:,k) = e_temp_std.^2;   %Matrix of squared errors (nonadaptive)
        e_b(i,:,k) = e_temp_b.^2;       %Matrix of squared errors (Benveniste)
        e_af(i,:,k) = e_temp_af.^2;     %Matrix of squared errors (Ang & Farhang)
        e_mx(i,:,k) = e_temp_mx.^2;     %Matrix of squared errors (Matthews & Xie)
        
        %Weight evolution
        w_std{i,k} = w_temp_std(1:end-1);   %Cell of filter weights (nonadaptive)
        w_b{i,k} = w_temp_b(1:end-1);       %Cell of filter weights (Benveniste)
        w_af{i,k} = w_temp_af(1:end-1);     %Cell of filter weights (Ang & Farhang)
        w_mx{i,k} = w_temp_mx(1:end-1);     %Cell of filter weights (Matthews & Xie)
    end
end

%% ENSEMBLE ERROR AVERAGE ACROSS REALISATIONS
e_std_mu1 = mean(10*log10(e_std(:,:,1)));   %Learning curve (nonadaptive) for learning rate: 0.01
e_std_mu2 = mean(10*log10(e_std(:,:,2)));   %Learning curve (nonadaptive) for learning rate: 0.1             
e_b_mu1 = mean(10*log10(e_b(:,:,1)));       %Learning curve (Benveniste) for learning rate: 0.01
e_b_mu2 = mean(10*log10(e_b(:,:,2)));       %Learning curve (Benveniste) for learning rate: 0.1
e_af_mu1 = mean(10*log10(e_af(:,:,1)));     %Learning curve (Ang & Farhang) for learning rate: 0.01
e_af_mu2 = mean(10*log10(e_af(:,:,2)));     %Learning curve (Ang & Farhang) for learning rate: 0.1
e_mx_mu1 = mean(10*log10(e_mx(:,:,1)));     %Learning curve (Matthews & Xie) for learning rate: 0.01
e_mx_mu2 = mean(10*log10(e_mx(:,:,2)));     %Learning curve (Matthews & Xie) for learning rate: 0.1

%% ENSEMBLE WEIGHT TRAJECTORY AVERAGE ACROSS REALISATIONS
w_std_mu1 = zeros(length(a), N); w_std_mu2 = zeros(length(a), N);
w_b_mu1 = zeros(length(a), N); w_b_mu2 = zeros(length(a), N);
w_af_mu1 = zeros(length(a), N); w_af_mu2 = zeros(length(a), N);
w_mx_mu1 = zeros(length(a), N); w_mx_mu2 = zeros(length(a), N);

for i=1:R
    w_temp_std1 = w_std{i,1}; w_temp_std2 = w_std{i,2};     %Nonadaptive
    w_temp_b1 = w_b{i,1}; w_temp_b2 = w_b{i,2};             %Benveniste
    w_temp_af1 = w_af{i,1}; w_temp_af2 = w_af{i,2};         %Ang & Farhang
    w_temp_mx1 = w_mx{i,1}; w_temp_mx2 = w_mx{i,2};         %Matthews & Xie
    
    w_std_mu1 = w_std_mu1 + w_temp_std1; w_std_mu2 = w_std_mu2 + w_temp_std2;
    w_b_mu1 = w_b_mu1 + w_temp_b1; w_b_mu2 = w_b_mu2 + w_temp_b2;
    w_af_mu1 = w_af_mu1 + w_temp_af1; w_af_mu2 = w_af_mu2 + w_temp_af2;
    w_mx_mu1 = w_mx_mu1 + w_temp_mx1; w_mx_mu2 = w_mx_mu2 + w_temp_mx2;
end

w_std_mu1 = w_std_mu1./R;  %Weights evolution for mu: 0.01 (nonadaptive)
w_std_mu2 = w_std_mu2./R;  %Weights evolution for mu: 0.1 (nonadaptive)
w_b_mu1 = w_b_mu1./R;      %Weights evolution for mu: 0.01 (Benveniste)
w_b_mu2 = w_b_mu2./R;      %Weights evolution for mu: 0.1 (Benveniste)
w_af_mu1 = w_af_mu1./R;    %Weights evolution for mu: 0.01 (Ang & Farhang)
w_af_mu2 = w_af_mu2./R;    %Weights evolution for mu: 0.1 (Ang & Farhang)
w_mx_mu1 = w_mx_mu1./R;    %Weights evolution for mu: 0.01 (Matthews & Xie)
w_mx_mu2 = w_mx_mu2./R;    %Weights evolution for mu: 0.1 (Matthews & Xie)

%% PLOT LEARNING CURVES AND WEIGHT ERROR
%Plot learning curves
figure(1); grid on; grid minor; hold on;
plot(1:N, e_std_mu1); plot(1:N, e_std_mu2);
plot(1:N, e_b_mu2);
plot(1:N, e_af_mu2);
plot(1:N, e_mx_mu2);
legend({['\mu_1=', num2str(mu(1)), '(Nonadaptive)'],['\mu_2=', num2str(mu(2)), '(Nonadaptive)'], ['\mu_2=', num2str(mu(2)), '(Benveniste)'], ['\mu_2=', num2str(mu(2)), '(Ang & Farhang)'],['\mu_2=', num2str(mu(2)), '(Matthews & Xie)']});
xlabel('Time Step'); ylabel('Squared Prediction Error (dB)');
title(['Learning Curve for LMS Forward Prediction on the MA(1) Process x(n) averaged over ', num2str(R), ' realisations (N=', num2str(N), ', a= ', num2str(a), ', \sigma_n^2=', num2str(sigma), ')']);

%Plot weight error curves
% step_start = [150, 250]*2;                            %Start of plateau region for time-averaging
% a_mu1 = round(mean(w_std_mu1(step_start(1):end)), 3);     %Converged estimate for a with mu: 0.01
% a_mu2 = round(mean(w_std_mu2(step_start(2):end)), 3);     %Converged estimate for a with mu: 0.1

figure(2); hold on; grid on; grid minor;
plot(1:N, a*ones(1,N) - w_std_mu1, 'LineWidth', 2); plot(1:N, a*ones(1,N) - w_std_mu2, 'LineWidth', 2);
plot(1:N, a*ones(1,N) - w_b_mu2, 'LineWidth', 2);
plot(1:N, a*ones(1,N) - w_af_mu2, 'LineWidth', 2);
plot(1:N, a*ones(1,N) - w_mx_mu2, 'LineWidth', 2); 
legend({['\mu_1=', num2str(mu(1)), '(Nonadaptive)'],['\mu_2=', num2str(mu(2)), '(Nonadaptive)'], ['\mu_2=', num2str(mu(2)), '(Benveniste)'], ['\mu_2=', num2str(mu(2)), '(Ang & Farhang)'],['\mu_2=', num2str(mu(2)), '(Matthews & Xie)']});
xlabel('Time Step'); ylabel('Weight Error (AU)'); xlim([length(a)+1, N]);
title('Evolution of LMS Adaptive Filter Coefficients against Time');