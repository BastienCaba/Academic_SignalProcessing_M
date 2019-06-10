%%% ASPMI Coursework 2019: ASSIGNMENT 2
%%% 2.2 Adaptive Step Size
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed

%% TASK A: Explore GASS algorithms
%PARAMETERS
R = 100;            %Number of realisations
N = 1000;           %Number of samples (signal length)
mu = [0.05, 0.1];   %Learning rate
mu0 = 0.1;          %Initial learning rate
a = 0.9;            %MA parameter
sigma = 0.5;        %MA noise parameter

rho = 5e-3;         %Hyperparameter for GASS learning rate update
alpha = 0.8;        %Hyperparameter for Ang & Farhang learning rate update

for i = 1:R         %Iterate over realisations
    [w_std1{i}, e_std1{i}] = fGASS(a, sigma, N, mu(1), rho, alpha, "standard");
    [w_std2{i}, e_std2{i}] = fGASS(a, sigma, N, mu(2), rho, alpha, "standard");
    [w_ben{i}, e_ben{i}] = fGASS(a, sigma, N, mu0, rho, alpha, "benveniste");
    [w_afa{i}, e_afa{i}] = fGASS(a, sigma, N, mu0, rho, alpha, "ang");
    [w_mxi{i}, e_mxi{i}] = fGASS(a, sigma, N, mu0, rho, alpha, "matthews");
end

%ENSEMBLE AVERAGES ACROSS REALISATIONS
Wstd1 = zeros(2,N); Wstd2 = zeros(2,N); Wben = zeros(2,N); Wafa = zeros(2,N); Wmxi = zeros(2,N);
Estd1 = zeros(1,N); Estd2 = zeros(1,N); Eben = zeros(1,N); Eafa = zeros(1,N); Emxi = zeros(1,N);
for i = 1:R
    w_t_std1 = w_std1{i}; w_t_std2 = w_std2{i}; w_t_ben = w_ben{i}; w_t_afa = w_afa{i}; w_t_mxi = w_mxi{i};
    e_t_std1 = e_std1{i}; e_t_std2 = e_std2{i}; e_t_ben = e_ben{i}; e_t_afa = e_afa{i}; e_t_mxi = e_mxi{i};
    
    Wstd1 = Wstd1 + w_t_std1; Wstd2 = Wstd2 + w_t_std2; Wben = Wben + w_t_ben; Wafa = Wafa + w_t_afa; Wmxi = Wmxi + w_t_mxi;
    Estd1 = Estd1 + e_t_std1; Estd2 = Estd2 + e_t_std2; Eben = Eben + e_t_ben; Eafa = Eafa + e_t_afa; Emxi = Emxi + e_t_mxi;
end
Wstd1 = Wstd1./R; Wstd2 = Wstd2./R; Wben = Wben./R; Wafa = Wafa./R; Wmxi = Wmxi./R;     %Mean Weights
Estd1 = Estd1./R; Estd2 = Estd2./R; Eben = Eben./R; Eafa = Eafa./R; Emxi = Emxi./R;     %Mean Error

%PLOT LEARNING CURVES AND WEIGHT ERROR
figure(1); subplot(1,2,1); grid on; grid minor; hold on;
plot(1:N, 10*log10(Estd1), 'LineWidth', 1);
plot(1:N, 10*log10(Estd2), 'LineWidth', 1); 
plot(1:N, 10*log10(Eben), 'LineWidth', 1); 
plot(1:N, 10*log10(Eafa), 'LineWidth', 1); 
plot(1:N, 10*log10(Emxi), 'LineWidth', 1);
legend({['Fixed \mu = ', num2str(mu(1))],['Fixed \mu = ', num2str(mu(2))], ['Benveniste (\mu_0 = ', num2str(mu0), ')'], ['Ang & Farhang (\mu_0 = ', num2str(mu0), ')'],['Matthews & Xie (\mu_0 = ', num2str(mu0), ')']});
xlabel('Time Step (AU)'); ylabel('Prediction Error Power (dB)');
title(['Comparison of the Learning Curve for LMS with Nonadaptive Learning Rate and GASS Algorithm (N=', num2str(N), ', a= ', num2str(a), ', \sigma_n^2=', num2str(sigma), ')']);

subplot(1,2,2); hold on; grid on; grid minor;
plot(1:N, a*ones(1,N) - Wstd1(1,:), 'LineWidth', 2);
plot(1:N, a*ones(1,N) - Wstd2(1,:), 'LineWidth', 2);
plot(1:N, a*ones(1,N) - Wben(1,:), 'LineWidth', 2);
plot(1:N, a*ones(1,N) - Wafa(1,:), 'LineWidth', 2);
plot(1:N, a*ones(1,N) - Wmxi(1,:), 'LineWidth', 2); 
legend({['Fixed \mu = ', num2str(mu(1))],['Fixed \mu = ', num2str(mu(2))], ['Benveniste (\mu_0 = ', num2str(mu0), ')'], ['Ang & Farhang (\mu_0 = ', num2str(mu0), ')'],['Matthews & Xie (\mu_0 = ', num2str(mu0), ')']});
xlabel('Time Step (AU)'); ylabel('Weight Error (AU)'); xlim([3 250]);
title('Evolution of Weight Error against Time Steps');