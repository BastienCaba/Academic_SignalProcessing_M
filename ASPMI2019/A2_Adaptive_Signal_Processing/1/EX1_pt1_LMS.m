%%% ASPMI Coursework 2019: ASSIGNMENT 2
%%% 2.1 The Least Mean Square (LMS) Algorithm
clear all; close all; clc;  %Initialise script
rng(1);                     %Set seed for random number generation

%% TASK B: Study the LMS algorithm
%PARAMETERS
R = 100;            %Number of realisations
N = 1000;           %Number of AR(2) samples
mu = [0.01, 0.05];  %Adaptation gain (learning rate)
a = [0.1, 0.8];     %AR(2) parameters
sigma = 0.25;       %AR(2) noise term

%Variables Initialisation
e = zeros(R, N, length(mu));    %Prediction error over realisations
e_sig = zeros(1,N);             %Error signal for one realisation
x = zeros(1,N);                 %Desired output (true AR process)
x_hat = zeros(1,N);             %Actual output (estimated AR process)
w_temp = randn(length(a), N);   %Filter weights (AR parameters)

%LMS FILTERS
for i = 1:R                     %Iterate over 100 realisations
    x(1:length(a)) = rand(1, length(a));                                %Initial condition for the signal x
    for k = 1:length(mu)        %Iterate over adaptation gain values
        for n = length(a)+1:N   %Iterate over time samples
            X = [x(n-1); x(n-2)];                                       %Input vector to LMS filter
            x(n) = a(1)*x(n-1) + a(2)*x(n-2) + sqrt(sigma)*randn(1);    %AR(2) definition (desired output)
            x_hat(:, n) = w_temp(:,n)'*X;                               %Predicted x (actual output)
            e_sig(n) = x(n) - x_hat(n);                                 %Prediction error
            w_temp(:,n+1) = w_temp(:,n) + mu(k).*e_sig(n).*X;           %Weight adjustment
        end
        e(i,:,k) = e_sig.^2;                                            %Squared error signal per realisation
        w{i,k} = w_temp(:, 1:end-1);                                    %Weight evolution per realisation
    end
end

%ENSEMBLE AVERAGES ACROSS REALISATIONS
e_mu1 = mean(e(:,:,1));                 %Prediction error power for learning rate: 0.01
e_mu2 = mean(e(:,:,2));                 %Prediction error power for learning rate: 0.05
e_mu1_log = mean(10*log10(e(:,:,1)));   %Learning curve for mu: 0.01
e_mu2_log = mean(10*log10(e(:,:,2)));   %Learning curve for mu: 0.05

w_mu1 = zeros(length(a), N);
w_mu2 = zeros(length(a), N);
for i=1:R
    w1 = w{i,1}; w_mu1 = w_mu1 + w1;
    w2 = w{i,2}; w_mu2 = w_mu2 + w2;
end
w_mu1 = w_mu1./R;                       %Weights evolution for mu: 0.01
w_mu2 = w_mu2./R;                       %Weights evolution for mu: 0.05

%Plot prediction error power for one realisation
figure(1); subplot(1,2,1); grid on; grid minor; hold on;
plot(1:N, 10*log10(e(1,:,1))); plot(1:N, 10*log10(e(1,:,2)));
legend({['\mu_1=', num2str(mu(1))],['\mu_2=', num2str(mu(2))]});
xlabel('Time Step (AU)'); ylabel('Error Power (dB)');
title(['Learning Curve for LMS Forward Prediction on 1 Realisation of the AR(2) Process x(n) (N=', num2str(N), ', a_1= ', num2str(a(1)), ', a_2=', num2str(a(2)), ', \sigma_n^2=', num2str(sigma), ')']);

%Plot learning curve for 100 realisations
subplot(1,2,2); grid on; grid minor; hold on;
plot(1:N, e_mu1_log); plot(1:N, e_mu2_log);
legend({['\mu_1=', num2str(mu(1))],['\mu_2=', num2str(mu(2))]});
xlabel('Time Step (AU)'); ylabel('Error Power (dB)');
title(['Learning Curve for LMS Forward Prediction on 100 Realisations of the AR(2) Process x(n) (N=', num2str(N), ', a_1= ', num2str(a(1)), ', a_2=', num2str(a(2)), ', \sigma_n^2=', num2str(sigma), ')']);

%% TASK C: Explore the misadjustment
step_start = [150, 250]*2;              %Start of plateau region for time-averaging
mse1 = mean(e_mu1(step_start(1):end));  %MSE for learning rate: 0.01
mse2 = mean(e_mu2(step_start(2):end));  %MSE for learning rate: 0.05
M1 = mse1/sigma - 1; M1t = mu(1)*50/54; %Misadjustment for learning rate: 0.01
M2 = mse2/sigma - 1; M2t = mu(2)*50/54; %Misadjustment for learning rate: 0.05

%% TASK D: Investigate weight convergence
a1_mu1 = round(mean(w_mu1(1, step_start(1):end)), 3);     %Converged estimate for a1 with mu: 0.01
a2_mu1 = round(mean(w_mu1(2, step_start(1):end)), 3);     %Converged estimate for a2 with mu: 0.01
a1_mu2 = round(mean(w_mu2(1, step_start(2):end)), 3);     %Converged estimate for a1 with mu: 0.05
a2_mu2 = round(mean(w_mu2(2, step_start(2):end)), 3);     %Converged estimate for a2 with mu: 0.05

%Plot Results
figure(2); hold on; grid on; grid minor;
plot(1:N, a(1)*ones(1,N), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2);
plot(1:N, a(2)*ones(1,N), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2);
plot(1:N, w_mu1(1,:), '--', 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1);
plot(1:N, w_mu1(2,:), '--', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1);
plot(1:N, w_mu2(1,:), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1);
plot(1:N, w_mu2(2,:), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1);
legend({['$a_1=0.1$'],['$a_2=0.8$'],['$\hat{a_1} \approx$', num2str(a1_mu1), ' ($\mu_1$=',num2str(mu(1)), ')'],['$\hat{a_2}\approx$', num2str(a2_mu1), ' ($\mu_1$=',num2str(mu(1)), ')'], ['$\hat{a_1}\approx$', num2str(a1_mu2), ' ($\mu_2$=',num2str(mu(2)), ')'], ['$\hat{a_2}\approx$', num2str(a2_mu2), ' ($\mu_2$=',num2str(mu(2)), ')']}, 'Interpreter', 'Latex', 'Location', 'Eastoutside');
xlabel('Time Step (AU)'); ylabel('Weight Value (AU)'); xlim([length(a)+1, N]);
title('Evolution of LMS Adaptive Filter Coefficients against Time Steps');