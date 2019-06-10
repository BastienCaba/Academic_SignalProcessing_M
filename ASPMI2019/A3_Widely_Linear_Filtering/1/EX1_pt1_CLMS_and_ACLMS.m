%%% ASPMI Coursework 2019: ASSIGNMENT 3
%%% 3.1 Widely Linear Filtering and Adaptive Spectrum Estimation
clear all; close all; clc;      %Initialise script

%% TASK A: WLMA(1)
%INITIALISATION: PARAMETERS
N = 1000;                       %Signal length (# samples)
R = 100;                        %Number of realisations
mu = 0.1;                       %Learning rate
b = [1.5 + 1i, 2.5 - 0.5*1i];   %WLMA(1) parameters
M = length(b);                  %Dimension of the design vector X

%INITIALISATION: VARIABLES
x = complex(zeros(1,N));        %CWGN
y = complex(zeros(1,N));        %WLMA(1) (true)
y_hat = complex(zeros(1,N));    %WLMA(1) (estimate)
e_a = complex(zeros(1,N));      %ACLMS error
e = complex(zeros(1,N));        %CLMS error
X = complex(zeros(M,N));        %Input Vector
algo = {'CLMS', 'ACLMS'};       %Name of algorithms

for j = 1:length(algo)          %Iterate over CLMS and ACLMS
    H = complex(zeros(M,N));    %Filter Weights H
    G = complex(zeros(M,N));    %Filter Weights G
    for i = 1:R                 %Iterate over realisations
        for n = 2:N
            x(n) = sqrt(1/2)*(randn(1) + 1i*randn(1));      %Circular complex WGN
            y(n) = x(n) + b(1)*x(n-1) + b(2)*conj(x(n-1));  %WLMA(1) process
            X(:,n) = [x(n); x(n-1)];                        %Input vector
            switch(algo{j})
                case 'CLMS'
                    [H(:,n+1), y_hat(n), e(i,n)] = clms(mu, X(:,n), y(n), H(:,n));
                case 'ACLMS'
                    [H(:,n+1), G(:,n+1), y_hat(n), e_a(i,n)] = aclms(mu, X(:,n), y(n), H(:,n), G(:,n));
            end
        end
    end
end

%Average errors over 100 realisations
error_clms = mean(e); sse_clms = mean(10*log10(abs(error_clms(500:end)).^2));
error_aclms = mean(e_a); sse_aclms = mean(10*log10(abs(error_aclms(500:end)).^2));

%PLOT RESULTS
figure(1); subplot(1,2,1); grid on; grid minor; hold on;
scatter(real(y), imag(y), 'filled'); scatter(real(x), imag(x), 'filled');
xlabel('Real'); ylabel('Imaginary'); title('Complex Scatter Plots for Circularity Check on WGN and WLMA(1)');
legend({'WLMA(1)', 'WGN'});

subplot(1,2,2); grid on; grid minor; hold on;
plot(10*log10(abs(error_clms).^2)); plot(10*log10(abs(error_aclms).^2));
xlabel('Time Step (AU)'); ylabel('Error Power (dB)'); title(['Learning Curves for CLMS and ACLMS Estimation of an WLMA(1) Process (b_1=', num2str(b(1)), ', b_2=', num2str(b(2)), ')']);
legend({'CLMS', 'ACLMS'}, 'Location', 'Southwest');

%% TASK B: Wind Data
clear all; clc;             %Reset
figure;
load('low-wind.mat');       %Load low-wind data
vel = v_east;               %East-West direction
vnl = v_north;              %North-South direction
zl = vel + j*vnl;           %Complex wind data

subplot(1,3,1);
rho = abs(mean((zl).^2)/mean(abs(zl).^2));  %Circularity coefficient
scatter(vel, vnl, 'filled');                %Circularity plot
xlabel('Real'); ylabel('Imaginary'); grid on; grid minor; legend({['\rho \approx ', num2str(round(rho,3))]});
title('Circularity Plot for Low Wind Regime');

load('medium-wind.mat');    %Load medium-wind data
vem = v_east;               %East-West direction
vnm = v_north;              %North-South direction
zm = vem + j*vnm;           %Complex wind data

subplot(1,3,2); 
rho = abs(mean((zm).^2)/mean(abs(zm).^2));  %Circularity coefficient
scatter(vem, vnm, 'filled');                %Circularity plot
xlabel('Real'); ylabel('Imaginary'); grid on; grid minor; legend({['\rho \approx ', num2str(round(rho,3))]});
title('Circularity Plot for Medium Wind Regime');

load('high-wind.mat');      %Load high-wind data
veh = v_east;               %East-West direction
vnh = v_north;              %North-South direction
zh = veh + j*vnh;           %Complex wind data

subplot(1,3,3);
rho = abs(mean((zh).^2)/mean(abs(zh).^2));  %Circularity coefficient
scatter(veh, vnh, 'filled');                %Circularity plot
xlabel('Real'); ylabel('Imaginary'); grid on; grid minor; legend({['\rho \approx ', num2str(round(rho,3))]}); 
title('Circularity Plot for High Wind Regime');

%COMPILE SIGNALS
z = {zl, zm, zh};
names = {'Low Wind Data', 'Medium Wind Data', 'High Wind Data'};

%% INITIALISATION: VARIABLES
N = 5000;                       %Length of wind data
R = 10;                         %Number of realisations
mu = [0.01, 0.005, 0.001];      %Learning rate
y_hat = complex(zeros(1,N));    %WLMA(1) (estimate)
e_a = complex(zeros(1,N));      %ACLMS error
e = complex(zeros(1,N));        %CLMS error

algo = {'CLMS', 'ACLMS'};
order_vec = 1:30;

%% ALGORITHM: CLMS v ACLMS
for p = 1:length(order_vec)
    order = order_vec(p);                   %Filter length
    M = order;                              %Dimension of the design vector X
    X = complex(zeros(M,N));                %Design Vector X
    for s = 1:length(z)
        y = z{s};                           %Wnd data
        for j = 1:length(algo)
            H = complex(zeros(M,N));        %Filter Weights H
            G = complex(zeros(M,N));        %Filter Weights G
            for i = 1:R
                for n = order+1:N
                    for k = 1:order
                        X(k,n) = y(n-k);
                    end
                    switch(algo{j})
                        case 'CLMS'
                            [H(:,n+1), y_hat(n), e(i,n)] = clms(mu(s), X(:,n), y(n), H(:,n));
                        case 'ACLMS'
                            [H(:,n+1), G(:,n+1), y_hat(n), e_a(i,n)] = aclms(mu(s), X(:,n), y(n), H(:,n), G(:,n));
                    end
                end
            end
        end
        error_clms(s,:) = mean(e); sse_clms(s, p) = 10*log10(mean(abs(error_clms(s,:).^2)));
        error_aclms(s,:) = mean(e_a); sse_aclms(s, p) = 10*log10(mean(abs(error_aclms(s,:).^2)));
    end
end

%PLOT RESULTS
figure; grid on; grid minor;
for i = 1:3
    subplot(1,3,i); hold on; plot(sse_clms(i,:), 'LineWidth', 2); plot(sse_aclms(i,:), 'LineWidth', 2);
    xlabel('Model Order'); ylabel('Mean Squared Error (dB)'); grid on; grid minor;
    title(['Mean Squared Error against Model Order for CLMS and ACLMS Prediction of ', names(i)]);
    legend({'CLMS', 'ACLMS'}); xlim([2 30]);
end