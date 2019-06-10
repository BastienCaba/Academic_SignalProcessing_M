%%% ASPMI Coursework 2019: ASSIGNMENT 2
%%% 2.1 The Least Mean Square (LMS) Algorithm
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed for random number generation

%% TASK E: Invetsigate Leaky LMS
%PARAMETERS
R = 100;                    %Number of realisations
N = 1000;                   %Number of AR(2) samples
mu = [0.01, 0.05];          %Adaptation gain (learning rate)
gamma = [0.1, 0.5, 0.9];    %Leakage coefficient
a = [0.1, 0.8];             %AR(2) parameters
sigma = 0.25;               %AR(2) noise term

%Variables Initialisation
e = zeros(R, N, length(mu));    %Prediction error over realisations
e_sig = zeros(1,N);             %Error signal for one realisation
x = zeros(1,N);                 %Desired output (true AR process)
x_hat = zeros(1,N);             %Actual output (estimated AR process)
w_temp = randn(length(a), N);   %Filter weights (AR parameters)

%Leaky LMS
for i = 1:R                     %Iterate over 100 realisations
    x(1:length(a)) = rand(1, length(a));                                                %Initial condition for the signal x
    for k = 1:length(mu)        %Iterate over adaptation gain values
        for j = 1:length(gamma) %Iterate over leakage coefficient values
            for n = length(a)+1:N   %Iterate over time samples
                X = [x(n-1); x(n-2)];                                                   %Input vector for LLMS
                x(n) = a(1)*x(n-1) + a(2)*x(n-2) + sqrt(sigma)*randn(1);                %AR(2) definition (desired output)
                x_hat(:, n) = w_temp(:,n)'*X;                                           %Predicted x (actual output)
                e_sig(n) = x(n) - x_hat(n);                                             %Prediction error
                w_temp(:,n+1) = (1-mu(k)*gamma(j))*w_temp(:,n) + mu(k).*e_sig(n).*X;    %Weight adjustment
            end
            e(i,:,k) = e_sig.^2;                                                        %Squared error signal per realisation
            w{i,k,j} = w_temp(:, 1:end-1);                                              %Weight evolution per realisation
        end
    end
end

%ENSEMBLE AVERAGES ACROSS REALISATIONS
w_mu1 = zeros(length(a), N, length(gamma));
w_mu2 = zeros(length(a), N, length(gamma));
step_start = [150, 250]*2;                                  %Start of plateau region for time-averaging

figure(1);
for j=1:length(gamma)
    for i=1:R
        w1 = w{i,1,j}; w_mu1(:,:,j) = w_mu1(:,:,j) + w1;
        w2 = w{i,2,j}; w_mu2(:,:,j) = w_mu2(:,:,j) + w2;
    end
    w_mu1(:,:,j) = w_mu1(:,:,j)./R;                         %Weights evolution for mu: 0.01
    w_mu2(:,:,j) = w_mu2(:,:,j)./R;                         %Weights evolution for mu: 0.05
    
    %WEIGHT CONVERGENCE
    a1_mu1 = round(mean(w_mu1(1, step_start(1):end,j)), 3);     %Converged estimate for a1 with mu: 0.01
    a2_mu1 = round(mean(w_mu1(2, step_start(1):end,j)), 3);     %Converged estimate for a2 with mu: 0.01
    a1_mu2 = round(mean(w_mu2(1, step_start(2):end,j)), 3);     %Converged estimate for a1 with mu: 0.05
    a2_mu2 = round(mean(w_mu2(2, step_start(2):end,j)), 3);     %Converged estimate for a2 with mu: 0.05
    
    std_a1_mu1(j) = round(std(w_mu1(1, step_start(1):end,j)), 3);     %Converged estimate for a1 with mu: 0.01
    std_a2_mu1(j) = round(std(w_mu1(2, step_start(1):end,j)), 3);     %Converged estimate for a2 with mu: 0.01
    std_a1_mu2(j) = round(std(w_mu2(1, step_start(2):end,j)), 3);     %Converged estimate for a1 with mu: 0.05
    std_a2_mu2(j) = round(std(w_mu2(2, step_start(2):end,j)), 3);     %Converged estimate for a2 with mu: 0.05
    
    %PLOT RESULTS
    subplot(1,length(gamma),j); hold on; grid on; grid minor;
    plot(1:N, a(1)*ones(1,N), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2);
    plot(1:N, a(2)*ones(1,N), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2);
    plot(1:N, w_mu1(1,:,j), '--', 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1);
    plot(1:N, w_mu1(2,:,j), '--', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1);
    plot(1:N, w_mu2(1,:,j), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1);
    plot(1:N, w_mu2(2,:,j), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1);
    legend({['$a_1=0.1$ (true)'],['$a_2=0.8$ (true)'],['$\hat{a_1} \approx$', num2str(a1_mu1), ' ($\mu_1$)'],['$\hat{a_2}\approx$', num2str(a2_mu1), ' ($\mu_1$)'], ['$\hat{a_1}\approx$', num2str(a1_mu2), ' ($\mu_2$)'], ['$\hat{a_2}\approx$', num2str(a2_mu2), ' ($\mu_2$)']}, 'Interpreter', 'Latex', 'Location', 'Eastoutside');
    xlabel('Time Step (AU)'); ylabel('Weight Value (AU)'); xlim([length(a)+1, N]);
    title(['Evolution of LMS Adaptive Filter Coefficients against Time (\gamma=', num2str(gamma(j)), ')']);
end