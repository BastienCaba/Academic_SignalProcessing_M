%%% ASPMI Coursework 2019: ASSIGNMENT 1
%%% 1.6 Robust Regression
clear all; close all; clc;  %Script initialisation

%% TASK: SINGULAR VALUE DECOMPOSITION
load('PCAPCR.mat');        	%Load DATA
figure(1); subplot(1,2,1); 	%Subplot indexing
svd_Xn = svd(Xnoise);    	%SINGLE VALUE DECOMPOSITION of the noise-corrupted data Xnoise
svd_X = svd(X);             %SINGLE VALUE DECOMPOSITION of the noise-free data X
bar([svd_X, svd_Xn]);       %Bar plot of eigenvalues
title('Singular Values of Noiseless Data X and Noise-Corrupted Data X_n');
xlabel('Singular Value Index'); ylabel('Singular Value Magnitude'); grid on; grid minor; legend({'Input X','Input+Noise X_n'});

%SQUARED ERROR computation and plot
error = (svd_X-svd_Xn).^2; subplot(1,2,2);          %Subplot indexing
bar(error); title('Squared Error between the Singular values of Input Data X and Input+Noise data X_n');
xlabel('Index'); ylabel("Squared Error, $\|x-\hat{x}\|^{2}$", 'Interpreter', 'Latex'); grid on; grid minor;

%% TASK: DIMENSIONALITY REDUCTION for the noise-corrupted data Xnoise
rank = 1:10;                            %Ranks to be trialed
for i = 1:length(rank)                  %Iterate over ranks
    r = rank(i);
    [Un, Sn, Vnoise] = svd(Xnoise);     %SVD for Xnoise
    Un_reduced = Un(:,1:r);             %Compute REDUCED matrix U for Xnoise
    Vn_reduced = Vnoise(:,1:r);         %Compute REDUCED matrix V for Xnoise
    Sn_reduced = Sn(1:r, 1:r);          %Compute REDUCED matrix S for Xnoise
    
    %RECONSTRUCT the REDUCED Xnoise matrix Xn_reduced
    Xn_reduced = Un_reduced*Sn_reduced*Vn_reduced';
    
    %COMPUTE ERRROR between data matrix X and matrices Xnoise and Xn_reduced
    noise_fullrank(i) = norm(X-Xnoise, 'fro');     %COMPUTE errror between X and Xnoise (noise in Xnoise)
    noise_subrank(i) = norm(X-Xn_reduced, 'fro');  %COMPUTE error between X and Xn_reduced (noise in Xn_reduced)
end

%Plot RESULTS
figure(2); plot(rank, noise_fullrank, 'LineWidth', 1); hold on; plot(rank, noise_subrank, 'LineWidth', 1);
title('Error Norm Across Variables when comparing Noiseless Data X to Noisy Data X_{noise} and Denoised Data X_{denoised}');
legend({'Error X-X_{noise}'}); xlabel('Rank r'); ylabel('$$\|\mathbf{X} - \tilde{\mathbf{X}}_{noise}\|_{F}$$', 'Interpreter', 'Latex'); grid on; grid minor;

%% TASK: Investigate the effect of the rank
r = 1:10;                               %Ranks to be trialed
e_PCR_train = zeros(length(r), 1);      %Error under PCR for training data
e_PCR_test = zeros(length(r), 1);       %Error under PCR for testing data

for i = 1:length(r)
    [Unoise, Snoise, Vnoise] = svd(Xnoise);     %SVD for Xnoise
    [Utest, Stest, Vtest] = svd(Xtest);         %SVD for Xtest
    
    %SUBRANK APPROXIMATION
    Xnoise_reduced = Unoise(:, 1:r(i)) * Snoise(1:r(i), 1:r(i)) * Vnoise(:, 1:r(i))';
    Xtest_reduced = Utest(:, 1:r(i)) * Stest(1:r(i), 1:r(i)) * Vtest(:, 1:r(i))';
    
    %WEIGHTS ESTIMATION (PCR and OLS SOLUTION for the REGRESSION MATRIX)
    B_PCR{i} = Vnoise(:, 1:r(i)) / Snoise(1:r(i), 1:r(i)) * Unoise(:, 1:r(i))' * Y;
    B_OLS = Xnoise'*Xnoise\Xnoise'*Y;
    
    %OUTPUT ESTIMATION
    Y_PCR = Xnoise_reduced * B_PCR{i};      %Output estimated using PCR on the training data
    Ytest_PCR = Xtest_reduced * B_PCR{i};   %Output estimated using PCR on the testing data
    Y_OLS = Xnoise*B_OLS;                   %Output estimated using OLS on the training data
    Ytest_OLS = Xtest*B_OLS;                %Output estimated using OLS on the testing data
    
    %ERRORS
    e_PCR_train(i) = norm(Y - Y_PCR, 'fro');
    e_PCR_test(i) = norm(Ytest - Ytest_PCR, 'fro');
    e_OLS_train(i) = norm(Y - Y_OLS, 'fro');
    e_OLS_test(i) = norm(Ytest - Ytest_OLS, 'fro');
end

%Plot RESULTS
figure; subplot(1,2,1); plot(r, e_PCR_train, 'LineWidth', 1); grid on; grid minor; hold on;
plot(r, e_OLS_train, 'LineWidth', 1); xlabel('Rank (r)'); legend({'PCR', 'OLS'});
ylabel('$$\|\textbf{Y} - \tilde{\textbf{Y}}\|_{F}$$', 'Interpreter', 'Latex'); xlim([1, 10]);
title('Evolution in the Training Error in Regressive Output Estimationas a function of Rank (r)');

subplot(1,2,2); plot(r, e_PCR_test, 'LineWidth', 1); grid on; grid minor; hold on;
plot(r, e_OLS_test, 'LineWidth', 1); xlabel('Rank (r)'); legend({'PCR', 'OLS'}); xlim([1, 10]);
ylabel('$$\|\textbf{Y}_{test} - \tilde{\textbf{Y}}_{test}\|_{F}$$', 'Interpreter', 'Latex');
title('Evolution in the Testing Error in Regressive Output Estimationas a function of Rank (r)');

%% TASK: Test algorithm on new testing data
figure; num_real = 100;                	%Number of realisations of testing data to be generated
error_PCR = zeros(num_real, length(r)); %Initialize L2 loss PCR error matrix
error_OLS = zeros(num_real, length(r)); %Initialize L2 loss OLS error matrix

for i = 1:num_real                      %Iterate over realisations
    for j = 1:length(r)                 %Iterate over ranks r
        [Y_test_OLS, Y_true_OLS] = regval(B_OLS);       %Apply OLS model to newly generated test data
        [Y_test_PCR, Y_true_PCR] = regval(B_PCR{j});    %Apply PCR model to newely generated test data
        error_PCR(i, j) = norm(Y_true_PCR - Y_test_PCR, 2)^2;   %Compute L2 Loss for PCR
        error_OLS(i, j) = norm(Y_true_OLS - Y_test_OLS, 2)^2;   %Compute L2 Loss for OLS
    end
    p1 = plot(error_OLS(i,:), 'Color', [0.9290, 0.6940, 0.1250]); p1.Color(4) = 0.1; hold on;
    p2 = plot(error_PCR(i,:), 'Color', [0, 0.4470, 0.7410]); p2.Color(4) = 0.1; hold on;
end

mu_PCR = mean(error_PCR);   %MSE under PCR
mu_OLS = mean(error_OLS);   %MSE under OLS
p3 = plot(mu_PCR, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2, "DisplayName", "PCR");
p4 = plot(mu_OLS, ':', 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2, "DisplayName", "OLS");
grid on; grid minor; legend([p3, p4]); xlabel('Rank (r)'); ylabel("$\|\mathbf{Y} - \tilde{\mathbf{Y}}\|_{2}^2$", 'Interpreter', 'Latex');
title(['L2 Loss and Mean Square Error as a function of Rank (r) Across ', num2str(num_real), ' Realisations of Testing Data for OLS and PCR']);