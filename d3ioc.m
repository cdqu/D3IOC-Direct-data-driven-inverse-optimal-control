% direct data-driven inverse optimal control
% author: Chendi Qu
% vanilla d3ioc


%% data gerneration
T_ini = 3;
N = 10;
n = 2;
m = 1;
p = 2;
% System Setup
sys_name = "double_integrator";
sys = get_system(sys_name);       

Q = [1,0.2;0.2,0.8];
R = eye(sys.nu) * 0.4;
q_r = Q;
r_r = R;
max_traj_len = 50;

% Simulation Using Random Input Sequence (Hankel matrix)
x_data = repmat(sys.xs, 1, T_ini+1);
u_data = repmat(sys.us, 1, T_ini);
y_data = repmat(sys.ys, 1, T_ini);
for i = 1:max_traj_len
    u = (-1 + rand(sys.nu, 1) * 2) * 0.1;
    u_data = [u_data, u];
    y_data = [y_data, sys.model.h(x_data(:, end), u)];
    x_data = [x_data, sys.model.f(x_data(:, end), u)];
end

% DeePC Controller Setup
fprintf("- Initializing DeePC Controllers ...\n");
data.u = u_data;
data.y = y_data;
params.Q = Q;
params.R = R;
params.T_ini = T_ini;
params.N = N;
params.n = n;
ldeepc_ctrl = DeePC(sys.constraints, sys.yf, sys.uf, params, data);
fprintf("\t Nominal DeePC initialized!\n");

% DeePC Simulation
fprintf("\n- Simulating DeePC (N = %d) ...\n", N);
x = repmat(sys.xs, 1, T_ini+1);
u = repmat(sys.us, 1, T_ini);
y = repmat(sys.ys, 1, T_ini);

[u_i, u_lq, info] = ldeepc_ctrl.solve(u(:, end-T_ini+1:end), y(:, end-T_ini+1:end));
for j = 1:N
    y = [y, sys.model.h(x(:, end), u_lq(j))];
    x = [x, sys.model.f(x(:, end), u_lq(j))];
end

%% ioc solver preparation

U = make_hankel(data.u, T_ini+N);
Y = make_hankel(data.y, T_ini+N);
Up = U(1:T_ini*m,:);
Uf = U(T_ini*m+1:end,:);
Yp = Y(1:T_ini*p,:);
Yf = Y(T_ini*p+1:end,:);

uo = u_lq';
y = y(:,T_ini+1:end);
yo = reshape(y,[],1);

q1 = sdpvar(1,1);
q2 = sdpvar(1,1);
q3 = sdpvar(1,1);
r1 = sdpvar(1,1);
Q = [q1,q2;q2,q3];
R = r1;
q_block = repmat({Q},1,N);
q_cal = blkdiag(q_block{:});
r_block = repmat({R},1,N);
r_cal = blkdiag(r_block{:});
alpha = sdpvar(1);
lambda = sdpvar((m+p)*T_ini,1);
omega_p_trans = [Up', Yp'];
theta = blkdiag(Q,R);
I = eye(size(theta));

%% feasible solution condition
uf_tilde = 0;
for i = 1:N
    uf_tilde = uf_tilde + kron(uo(i),2.*Uf(i,:)');
end
yf_tilde = 0;
for j = 1:N
    yf_tilde = yf_tilde + kron(y(:,j)',2.*Yf((j-1)*n+1:j*n,:)');
end
phi = [omega_p_trans, uf_tilde, yf_tilde];
rank(phi,1e-6)
phi_v = [phi(:,1:end-3), phi(:,end-2)+phi(:,end-1), phi(:,end)];
rank(phi_v,1e-6)

%% QP-SDP without noises

cons = [omega_p_trans * lambda + 2.* Uf'* r_cal* uo + ...
    2.* Yf'* q_cal* yo == 0, theta >= I,theta <= alpha.*I];
ops = sdpsettings('solver', 'sedumi', 'verbose',1);
z = alpha*alpha;

result = solvesdp(cons,z);
if result.problem == 0
    double(Q)
    double(R)
    double(lambda)
    double(alpha)
else
    disp('WRONG');
    result.info
end

%% with noise

vec = [lambda;q1;q2;q3;r1];
cons = [norm(vec)^2 == 10,R>=0,Q>=0];
ops = sdpsettings('solver','bmibnb','bmibnb.uppersolver','fmincon');
% options=optimset('TolCon',1e-3);%6

res = omega_p_trans * lambda + 2.* Uf'* r_cal* uo + 2.* Yf'* q_cal* yo;
z = res'*res;

result = solvesdp(cons,z,ops);
if result.problem == 0 
    double(Q)
    double(R)
    double(z)
else
    disp('WRONG');
    result.info
end

%% estimation error

qr = blkdiag(double(Q),double(R));
qr_real = blkdiag(q_r, r_r);

objective = @(t) norm(qr * t - qr_real, 'fro') / norm(qr_real, 'fro');
t0 = 1;  % Initial guess for scalar t
options = optimset('Display', 'iter');  % Display iteration details
[t_optimal, es_err] = fminunc(objective, t0, options);
