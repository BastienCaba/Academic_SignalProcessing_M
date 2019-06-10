function [w, e] = fGASS(a, sigma, N, mu0, rho, alpha, label)
%% INITIALISATION
eta = zeros(1,N);                   %Driving noise (WGN)
x = zeros(1,N);                     %True signal
x_hat = zeros(1,N);                 %Signal prediction
e_t = zeros(1,N);                   %Error signal
w_t = randn(length(a)+1, N);        %Filter weights
X = zeros(length(a)+1,N);           %Design vector
if(label ~= "standard")             %If GASS is implemented
    psi = zeros(length(a)+1, N);    %Initialise PSI
end
mu(1:3) = mu0*ones(1,3);            %Initialise learning rate
for i = 1:length(a)+2
    x(i) = rand(1);                 %Initialise the true signal x
    eta(i) = sqrt(sigma).*randn(1); %Initiaise the noise term eta
    if(label ~= "standard")
        psi(:,i) = rand(1);
    end
end

for n = length(a)+2:N
    eta(n) = sqrt(sigma)*randn(1);  %Generate noise sample
    x(n) = a*eta(n-1) + eta(n);     %MA(1) Definition of the true signal
    X(:,n) = [eta(n-1); eta(n)];    %Design vector
    
    x_hat(n) = w_t(:,n)'*X(:,n);    %Predicted value
    e_t(n) = x(n) - x_hat(n);       %Error
    w_t(:,n+1) = w_t(:,n) + mu(n).*e_t(n)*X(:,n);  %Weight update
    
    if(label ~= "standard")
        switch label
            case "benveniste"
                psi(:,n) = (eye(length(a)+1)-mu(n-1)*X(:,n-1)*X(:,n-1)')*psi(:,n-1) + e_t(n-1)*X(:,n-1);
            case "ang"
                psi(:,n) = alpha*psi(:,n-1) + e_t(n-1)*X(:,n-1);
            case "matthews"
                psi(:,n) = e_t(n-1)*X(:,n-1);
        end
        mu(n+1) = mu(n) + rho*e_t(n)*X(:,n)'*psi(:,n);
        
    elseif(label == "standard")
        mu(n+1) = mu(n);
    end
end

e = e_t;                %Matrix of squared errors (nonadaptive)
w = w_t(:,1:end-1);     %Cell of filter weights (nonadaptive)
end