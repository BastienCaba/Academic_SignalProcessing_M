%%% ASPMI Coursework 2019: ASSIGNMENT 2
%%% 2.2 Adaptive Step Size
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed

%% TASK C: Implement GNGD
%PARAMETERS
R = 250;                    %Number of realisations
N = 1000;                   %Number of samples (signal length)
a = 0.9;                    %MA parameter
sigma = 0.5;                %MA noise parameter

rho_eps = [0.5, 0.7];       %Hyperparameter for learning epsilon (NLMS with GNGD)
mu0_eps = 1;                %Initial learning rate (GNGD)
rho_ben = [0.01, 0.015];    %Hyperparameter for learning mu (Benveniste)
mu0_ben = 0.1;              %Initial learning rate (Benveniste)
alpha = 0.8;                %Hyperparameter for Ang & Farhang (not required)

for i = 1:R                 %Iterate over realisations
    [w{i}, e{i}] = fGNGD(a, sigma, N, rho_eps(1), mu0_eps);
    [wu{i}, eu{i}] = fGNGD(a, sigma, N, rho_eps(2), mu0_eps);
    [w_ben{i}, e_ben{i}] = fGASS(a, sigma, N, mu0_ben, rho_ben(1), alpha, "benveniste");
    [w_benu{i}, e_benu{i}] = fGASS(a, sigma, N, mu0_ben, rho_ben(2), alpha, "benveniste");
end

%ENSEMBLE AVERAGES ACROSS REALISATIONS
W = zeros(2,N); Wben = zeros(2,N); Wu = zeros(2,N); Wbenu = zeros(2,N); 
E = zeros(1,N); Eben = zeros(1,N);
for i = 1:R
    w_t = w{i}; w_tben = w_ben{i};
    w_tu = wu{i}; w_tbenu = w_benu{i};
    e_t = e{i}; e_tben = e_ben{i};
    W = W + w_t; Wben = Wben + w_tben;
    Wu = Wu + w_tu; Wbenu = Wbenu + w_tbenu;
    E = E + e_t; Eben = Eben + e_tben;
end
W = W./R; Wben = Wben./R; 
Wu = Wu./R; Wbenu = Wbenu./R; 
E = E./R; Eben = Eben./R; 

%PLOT LEARNING CURVES AND WEIGHT ERROR
figure(1); grid on; grid minor; hold on;
p1 = plot(1:N, a*ones(1,N) - Wu(1,:), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2); p1.Color(4) = 0.3;
p2 = plot(1:N, a*ones(1,N) - Wbenu(1,:), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2); p2.Color(4) = 0.3;
plot(1:N, a*ones(1,N) - W(1,:), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2); 
plot(1:N, a*ones(1,N) - Wben(1,:), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2); 
legend({['GNGD (\mu_0=', num2str(mu0_eps), ', \rho=', num2str(rho_eps(2)), ')'],['Benveniste (\mu_0=', num2str(mu0_ben), ', \rho=', num2str(rho_ben(2)), ')'], ['GNGD (\mu_0=', num2str(mu0_eps), ', \rho=', num2str(rho_eps(1)), ')'],['Benveniste (\mu_0=', num2str(mu0_ben), ', \rho=', num2str(rho_ben(1)), ')']});
xlabel('Time Step (AU)'); ylabel('Weight Error (AU)'); xlim([length(a)+1, N]);
title('Evolution of Weight Error for LMS with GASS (Benveniste) and NLMS with GNGD against Time Steps'); xlim([3 80]); ylim([-0.2 1.5]);