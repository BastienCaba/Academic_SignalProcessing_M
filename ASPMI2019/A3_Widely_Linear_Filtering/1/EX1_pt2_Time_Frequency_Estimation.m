%%% ASPMI Coursework 2019: ASSIGNMENT 3
%%% 3.1 Widely Linear Filtering and Adaptive Spectrum Estimation
clear all; close all; clc;      %Initialise script

%% TASK C: Power Systems
M = 3;                          %Number of phases
f0 = 50;                        %System frequency (in Hz)
fs = 5000;                      %Sampling frequency (in Hz)
N = 1500;                       %Number of samples
t = 0:N;                        %Time axis (in s)

V = [1,1; 1,2; 1,3];                    %Magnitudes
Vlabel = {'Balanced', 'Unbalanced'};    %Magnitude labels
D = [0,0; 0,pi/2; 0,-pi/2];             %Phase distortion
Dlabel = {'Balanced', 'Unbalanced'};    %Phase distortion labels
phi = [0; 2*pi/3; -2*pi/3];             %Relative phase shift
shift = zeros(M,1);                     %Absolute phase shift

%Clarke Matrix Definition
C = sqrt(2/3)*[(sqrt(2)/2)*ones(1,3); 1 -1/2 -1/2; 0 sqrt(3)/2 -sqrt(3)/2 ];

figure; grid on; grid minor; hold on;
for i = 1:length(V(1,:))
    for j = 1:length(D(1,:))
        v{i,j} = V(:,i) .* cos(2 * pi * (f0 / fs) * t + shift + D(:,j) + phi);
        v_clarke = C*v{i,j};                                    %Clarke transform
        v_complex(i,j,:) = v_clarke(2,:) + 1j*v_clarke(3,:);    %Complex voltages (alpha-beta)
        rho = round(abs(mean((v_complex(i,j,:)).^2)/mean(abs(v_complex(i,j,:)).^2)), 2);
        s = scatter(real(v_complex(i,j,:)), imag(v_complex(i,j,:)), 15, 'filled');
        set(s, {'DisplayName'}, {['V: ', Vlabel{i}, ', \Delta: ', Dlabel{j}, ', \rho \approx ', num2str(rho)]});
    end
end
legend show; xlabel('Real'); ylabel('Imaginary'); axis equal;
title('Circularity Plot of Balanced and Unbalanced Three-Phase \alpha-\beta Voltage Systems');

%% TASK E: Frequency Analysis
%INITIALISATION: VARIABLES
y = complex(zeros(1,N));        %WLMA(1) (true)
y_hat = complex(zeros(1,N));    %WLMA(1) (estimate)
e_a = complex(zeros(1,N));      %ACLMS error
e = complex(zeros(1,N));        %CLMS error
X = complex(zeros(1,N));        %Design Vector

M = 1;                          %Dimension of design vector X
R = 100;                        %Number of realisations
mu = [0.05, 0.01];              %Learning rate
signal = [[1 1];[2 2]];         %Index of the purely balanced and purely unbalanced Clarke voltages
algo = {'CLMS', 'ACLMS'};       %Algorithm names
sig_names = {'Balanced', 'Unbalanced'};

for i = 1:length(sig_names)
for j = 1:length(algo) 
    H = complex(zeros(M,N));    %Filter Weights H
    G = complex(zeros(M,N));    %Filter Weights G
    for r = 1:R
        for n = 2:N
            y(n) = v_complex(signal(i,1),signal(i,2),n);
            X(:,n) = y(n-1);
            switch(algo{j})
                case 'CLMS'
                    [H(:,n+1), y_hat(n), e(r,n)] = clms(mu(i), X(:,n), y(n), H(:,n));
                case 'ACLMS'
                    [H(:,n+1), G(:,n+1), y_hat(n), e_a(r,n)] = aclms(mu(i), X(:,n), y(n), H(:,n), G(:,n));
            end
        end
        switch(algo{j})
            case 'CLMS'
                H_CLMS{r} = H(:,2:end);
            case 'ACLMS'
                H_ACLMS{r} = H(:,2:end);
                G_ACLMS{r} = G(:,2:end);
        end
    end
end

%Average errors over 100 realisations
error_clms = 10*log10(mean(abs(e).^2));
error_aclms = 10*log10(mean(abs(e_a).^2));

%Average weights over 100 realisations
H_mean_CLMS = zeros(M,N);
H_mean_ACLMS = zeros(M,N);
G_mean_ACLMS = zeros(M,N);
for p = 1:R
    H_mean_CLMS = H_mean_CLMS + H_CLMS{p};
    H_mean_ACLMS = H_mean_ACLMS + H_ACLMS{p};
    G_mean_ACLMS = G_mean_ACLMS + G_ACLMS{p};
end
H_mean_CLMS = H_mean_CLMS ./ R;
H_mean_ACLMS = H_mean_ACLMS ./ R;
G_mean_ACLMS = G_mean_ACLMS ./ R;

%Plot learning curves
figure; subplot(1,2,1); grid on; grid minor; hold on;
plot(error_clms, 'LineWidth', 2); plot(error_aclms, 'LineWidth', 2);
sse_clms(i) = mean(error_clms(500:end)); sse_aclms(i) = mean(error_aclms(1200:end));
xlabel('Time Steps (AU)'); ylabel('Error Power (dB)'); legend({'CLMS', 'ACLMS'});
title(['Learning Curves for CLMS and ACLMS on ', sig_names{i}, ' Clarke Voltage Data (avergaged over ', num2str(R), ' realisations)']);

%Plot weight evolution
fo_CLMS = abs(fs/(2*pi)*atan(imag(H_mean_CLMS.')./real(H_mean_CLMS.')));
fo_ACLMS = abs(fs/(2*pi)*atan(sqrt(imag(H_mean_ACLMS.').^2 - abs(G_mean_ACLMS.').^2)./real(H_mean_ACLMS.')));
subplot(1,2,2); grid on; grid minor; hold on; 
plot(fo_CLMS, 'LineWidth', 1); plot(fo_ACLMS, 'LineWidth', 1); plot(f0*ones(1,N), 'k--', 'Linewidth', 1);
xlabel('Time Steps (AU)'); ylabel('$$\hat{f_0}$$ (Hz)', 'Interpreter', 'Latex'); legend({'CLMS', 'ACLMS', 'f_0 = 50Hz'});
title(['Evolution of the Estimate of the System Frequency f_0 (avergaged over ', num2str(R), ' realisations)']);
ylim([0 150]); xlim([0 1000]);
end