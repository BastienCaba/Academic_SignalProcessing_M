function [w, e] = fGNGD(a, sigma, N, rho, mu0)
%% INITIALISATION
eta = zeros(1,N);                   %Driving noise (WGN)
x = zeros(1,N);                     %True signal
x_hat = zeros(1,N);                 %Signal prediction
e_t = rand(1,N);                    %Error signal
w_t = randn(length(a)+1, N);        %Filter weights
X = zeros(length(a)+1,N);           %Design vector
epsilon = zeros(1,N);               %Regularization factor
mu_mod = zeros(1,N);                %Modified learing rate

beta = 1;

for i = 1:length(a)+2
    x(i) = rand(1);                 %Initialise the true signal x
    eta(i) = sqrt(sigma).*randn(1); %Initiaise the noise term eta
    epsilon(i) = 1/mu0;
end

for m = 1:length(a)+1
    X(m,length(a)+1) = eta(length(a)+2-m);
end

for n = length(a)+2:N
    eta(n) = sqrt(sigma)*randn(1);  %Generate noise sample
    x(n) = a*eta(n-1) + eta(n);     %MA(1) Definition of the true signal
    for k = 1:length(a)+1
        X(k,n) = eta(n-k+1);
    end
    X(:,n) = [eta(n-1); eta(n)];    %Design vector
    
    mu_mod(n) = beta/(epsilon(n) + X(:,n)'*X(:,n));
    
    x_hat(n) = w_t(:,n)'*X(:,n);    %Predicted value
    e_t(n) = x(n) - x_hat(n);       %Error
    w_t(:,n+1) = w_t(:,n) + mu_mod(n)*e_t(n)*X(:,n);  %Weight update
    
    epsilon(n+1) = epsilon(n) - rho * mu0 * ((e_t(n)*e_t(n-1)*X(:,n)'*X(:,n-1)) / (epsilon(n-1) + X(:,n-1)'*X(:,n-1))^2);
end

e = e_t;                %Matrix of squared errors (nonadaptive)
w = w_t(:,1:end-1);     %Cell of filter weights (nonadaptive)
end