clc; clear; close all;

rng(40);

% Simulation Parameters
dt = 0.1; % Time step [s]
T = 10; % Total simulation time [s]
time = 0:dt:T; % Time vector
N = length(time); % Number of steps
g = 9.81; % Gravity acceleration
% noisy
gyro_noise = 1;
acc_noise = 0.2;

% Define True Euler Angles (Cardan Angles)
phi = 10 * sind(10 * time);  % Roll (sinusoidal oscillation)
theta = 5 * cosd(5 * time);  % Pitch (cosinusoidal oscillation)
psi = 20 * sind(3 * time);   % Yaw (slower variation)
x_true = [phi; theta; psi];

% Plot Trajectory
point_body = [1; 0.5; 0.4]; % A point of the body frame
trajectory = zeros(3, N);
for ii = 1:N
    % Compute rotation matrix
    Rot = eul2rotm(deg2rad([phi(ii), theta(ii), psi(ii)]), 'XYZ');
    trajectory(:,ii) = Rot * point_body; % Transform point into inertial frame
end
figure;
plot3(trajectory(1,:), trajectory(2,:), trajectory(3,:), 'k', 'LineWidth', 2, 'LineStyle',':');
hold on;
scatter3(trajectory(1,1), trajectory(2,1), trajectory(3,1), 100, 'go', 'filled'); % Start point
scatter3(trajectory(1,end), trajectory(2,end), trajectory(3,end), 100, 'ro', 'filled'); % End point
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Trajectory of a Rotating Point in Inertial Frame', 'FontSize', 14);
legend('Trajectory','Start','End', Location='best', fontsize=14);
grid on;
view(3);

% Compute Angular Velocities from Euler Angles
p = 100 * cosd(10 * time); % Roll rate
q = -25 * sind(5 * time);  % Pitch rate
r = 60 * cosd(3 * time);   % Yaw rate
w_true = [p; q; r];
w_noisy = w_true + gyro_noise * randn(size(w_true));

% Simulate Accelerometer Measurements
a = -g * sind(theta); % Acceleration along X
b = g * sind(phi) .* cosd(theta); % Acceleration along Y
c = g * cosd(phi) .* cosd(theta); % Acceleration along Z
phi_acc = atan2d(b, c);
theta_acc = atan2d(-a, sqrt(b.^2 + c.^2));
xacc_true = [phi_acc; theta_acc];
xacc_noisy = xacc_true + acc_noise * randn(size(xacc_true));

% Plot the Cardan Angles
figure;
subplot(3,1,1); 
plot(time, phi, 'r', 'LineWidth', 3); 
hold on;
scatter(time, xacc_noisy(1,:), 10, 'b', 'filled');
ylabel('Roll \phi [deg]', 'FontSize', 14);
title('Roll angle', 'FontSize', 14);
legend('True angle \phi','Recorded \phi from sensor', Location='best', fontsize=14)

subplot(3,1,2); 
plot(time, theta, 'g', 'LineWidth', 3); 
hold on;
scatter(time, xacc_noisy(2,:), 10, 'k', 'filled');
ylabel('Pitch \theta [deg]', 'FontSize', 14);
title('Pitch angle', 'FontSize', 14);
legend('True angle \theta','Recorded \theta from sensor', Location='best', fontsize=14)

subplot(3,1,3); 
plot(time, psi, 'b', 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 14); ylabel('Yaw \psi [deg]', 'FontSize', 14);
title('Yaw angle', 'FontSize', 14);
legend('True angle \psi', Location='best', fontsize=14)

% Plot the angular velocity
figure;
subplot(3,1,1); 
plot(time, p, 'r', 'LineWidth', 2); hold on;
scatter(time, w_noisy(1,:), 10, 'b', 'filled');
ylabel('p [deg/s]', 'FontSize', 14);
title('Roll angular velocity', 'FontSize', 14);
legend('True angular velocity p', 'Recorded p from sensor', Location='best', fontsize=14)

subplot(3,1,2); 
plot(time, q, 'g', 'LineWidth', 2); hold on;
scatter(time, w_noisy(2,:), 10, 'k', 'filled');
ylabel('q [deg/s]', 'FontSize', 14);
title('Pitch angular velocity', 'FontSize', 14);
legend('True angular velocity q', 'Recorded q from sensor', Location='best', fontsize=14)

subplot(3,1,3); 
plot(time, r, 'b', 'LineWidth', 2); hold on;
scatter(time, w_noisy(3,:), 10, 'm', 'filled');
xlabel('Time [s]', 'FontSize', 14); ylabel('r [deg/s]', 'FontSize', 14);
title('Yaw angular velocity', 'FontSize', 14);
legend('True angular velocity r', 'Recorded r from sensor', Location='best', fontsize=14)

%% EKF Initialization

% Initial state and covariance
m_k = [0; 5; 0]; % Initial guess for angles [phi, theta, psi]
P_k = diag([1 1 1]); % Initial covariance

% Store results
m_est = zeros(3, N);
m_est(:,1) = m_k;

% State transition function
f = @(x, omega) [ ...
    x(1) + dt * (omega(1) + sind(x(1)) * tand(x(2)) * omega(2) + cosd(x(1)) * tand(x(2)) * omega(3));
    x(2) + dt * (cosd(x(1)) * omega(2) - sind(x(1)) * omega(3));
    x(3) + dt * (sind(x(1)) / cosd(x(2)) * omega(2) + cosd(x(1)) / cosd(x(2)) * omega(3))
];

% Measurement function
h = @(x) [x(1); x(2)];

% Jacobians
F_x = @(x, omega) [ ...
    1 + dt * (cosd(x(1)) * tand(x(2)) * omega(2) - sind(x(1)) * tand(x(2)) * omega(3)), ...
    dt * (sind(x(1)) * (1/cosd(x(2))^2) * omega(2) + cosd(x(1)) * (1/cosd(x(2))^2) * omega(3)), ...
    0;

    dt * (-sind(x(1)) * omega(2) - cosd(x(1)) * omega(3)), ...
    1, ...
    0;

    dt * (cosd(x(1)) / cosd(x(2)) * omega(2) - sind(x(1)) / cosd(x(2)) * omega(3)), ...
    dt * (sind(x(1)) * sind(x(2)) / cosd(x(2))^2 * omega(2) + cosd(x(1)) * sind(x(2)) / cosd(x(2))^2 * omega(3)), ...
    1
];

H_x = @(x) [1, 0, 0; 0, 1, 0];

% Kalman loop
for ii = 2:N
    % Prediction step
    m_k_minus = f(m_k, w_noisy(:,ii-1));
    P_k_minus = F_x(m_k, w_noisy(:,ii-1)) * P_k * F_x(m_k, w_noisy(:,ii-1))' + gyro_noise*eye(3);

    % Innovation (residual)
    v_k = xacc_noisy(:,ii) - h(m_k_minus);
    S_k = H_x(m_k_minus) * P_k_minus * H_x(m_k_minus)' + acc_noise*eye(2);

    % Kalman gain
    K_k = P_k_minus * H_x(m_k_minus)' / S_k;

    % Update step
    m_k = m_k_minus + K_k * v_k;
    P_k = (eye(3) - K_k * H_x(m_k_minus)) * P_k_minus;

    % Store estimates
    m_est(:, ii) = m_k;
end

rmse_phi = sqrt(mean((x_true(1,:) - m_est(1,:)).^2));
fprintf('RMSE for phi: %.4f\n', rmse_phi);

rmse_theta = sqrt(mean((x_true(2,:) - m_est(2,:)).^2));
fprintf('RMSE for theta: %.4f\n', rmse_theta);

rmse_psi = sqrt(mean((x_true(3,:) - m_est(3,:)).^2));
fprintf('RMSE for psi: %.4f\n', rmse_psi);

figure;
subplot(3,1,1);
plot(time, x_true(1,:), 'r', 'LineWidth', 1.5); hold on;
plot(time, m_est(1,:), 'b', 'LineWidth', 1.2);
title('EKF Estimate - Roll Angle \phi', 'FontSize', 14);
legend('True', 'EKF Estimate',Location='best', fontsize=14);
ylabel('\phi [deg]', 'FontSize', 14);

subplot(3,1,2);
plot(time, x_true(2,:), 'g', 'LineWidth', 1.5); hold on;
plot(time, m_est(2,:), 'k', 'LineWidth', 1.2);
title('EKF Estimate - Pitch Angle \theta', 'FontSize', 14);
legend('True', 'EKF Estimate',Location='best', fontsize=14);
ylabel('\theta [deg]', 'FontSize', 14);

subplot(3,1,3);
plot(time, x_true(3,:), 'b', 'LineWidth', 1.5); hold on;
plot(time, m_est(3,:), 'm', 'LineWidth', 1.2);
title('EKF Estimate - Yaw Angle \psi', 'FontSize', 14);
legend('True', 'EKF Estimate',Location='best', fontsize=14);
xlabel('Time [s]', 'FontSize', 14); ylabel('\psi [deg]', 'FontSize', 14);
