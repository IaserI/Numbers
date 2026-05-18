% Gruppo N
% Giovanni Bozzi, Tommaso Manca


clear;
clc;

F = @(x) [
    10*(x(1)^2 - x(2)^2) + x(1) - 0.5;
    10*(x(2)^2 - x(3)^2) + x(2) - x(1);
    10*(x(3)^2 - x(4)^2) + x(3) - x(2);
    5*(x(4)^2 - 0.04) + x(4) - 0.3
];

F1 = @(x) [
    20*x(1) + 1,  -20*x(2),       0,              0;
    -1,            20*x(2) + 1,   -20*x(3),       0;
    0,             -1,            20*x(3) + 1,   -20*x(4);
    0,              0,            0,              10*x(4) + 1
];

iter_max = 66;
tol = [1e-9, 1e-9];
x0_list = [zeros(4, 1), ones(4, 1)];
method_names = {'corde', 'Newton'};
method_m = [0, 1];

if exist('fsolve', 'file') == 0
    error('esercizio4:MissingFsolve', ...
        'fsolve richiede l''Optimization Toolbox di MATLAB.');
end

if exist('optimoptions', 'file') ~= 0
    options = optimoptions('fsolve', 'Display', 'off');
else
    options = optimset('Display', 'off');
end
x_star = fsolve(F, ones(4, 1), options);
norm_x_star = norm(x_star, inf);

fprintf('Soluzione di riferimento fsolve:\n');
fprintf('  %.12e %.12e %.12e %.12e\n\n', x_star);

results = cell(size(x0_list, 2), length(method_m));

for idx_x0 = 1:size(x0_list, 2)
    x0 = x0_list(:, idx_x0);

    figure;
    hold on;

    for idx_method = 1:length(method_m)
        m_method = method_m(idx_method);
        [x, iter, normF_rec, condF1_rec] = ...
            newtons(x0, F, F1, iter_max, tol, m_method);

        final_rel_err = norm(x - x_star, inf) / norm_x_star;

        fprintf('Innesco %d, metodo %s: iter = %d, errore relativo finale = %.10e\n', ...
            idx_x0, method_names{idx_method}, iter, final_rel_err);

        valid_plot = isfinite(normF_rec) & normF_rec > 0;
        if any(valid_plot)
            semilogy(find(valid_plot) - 1, normF_rec(valid_plot), ...
                '-o', 'LineWidth', 1.2);
        else
            semilogy(NaN, NaN, '-o', 'LineWidth', 1.2);
        end

        results{idx_x0, idx_method} = struct( ...
            'x', x, 'iter', iter, ...
            'normF_rec', normF_rec, 'condF1_rec', condF1_rec);
    end

    grid on;
    title(sprintf('Norma infinito di F da x0 = (%g,%g,%g,%g)^T', x0));
    xlabel('Iterazione');
    ylabel('||F(x_k)||_\infty');
    legend(method_names, 'Location', 'best');
end

newton_from_ones = results{2, 2};
figure;
plot(0:newton_from_ones.iter, newton_from_ones.condF1_rec, ...
    '-o', 'LineWidth', 1.2);
title('Condizionamento della Jacobiana - Newton da x0 = (1,1,1,1)^T');
xlabel('Iterazione');
ylabel('cond_\infty(J_F(x_k))');
grid on;
legend('Newton', 'Location', 'best');




% function
function [x, iter, normF_rec, condF1_rec] = newtons(x, F, F1, iter_max, tol, m)

    % controlli sugli input
    if ~isa(F, 'function_handle') || ~isa(F1, 'function_handle')
        error('newtons:InvalidHandles', ...
            'F e F1 devono essere function handle.');
    end

    if ~isscalar(iter_max) || iter_max <= 0 || iter_max ~= floor(iter_max)
        error('newtons:InvalidIterMax', ...
            'iter_max deve essere un intero positivo.');
    end

    if numel(tol) ~= 2 || any(tol <= 0)
        error('newtons:InvalidTolerance', ...
            'tol deve contenere due tolleranze positive.');
    end

    if ~isscalar(m)
        error('newtons:InvalidMethodFlag', 'm deve essere uno scalare.');
    end

    x = x(:);
    tau_a = tol(1);
    tau_r = tol(2);

    % cache di F e F1: ad ogni iterazione li valutiamo una sola volta
    % (per le registrazioni) e li riutilizziamo nel solve dell'iterazione
    % successiva.
    F_cached  = F(x);
    F1_cached = F1(x);
    F0_norm   = norm(F_cached, inf);
    threshold = tau_a + tau_r * F0_norm;

    normF_rec  = zeros(iter_max + 1, 1);
    condF1_rec = zeros(iter_max + 1, 1);
    normF_rec(1)  = F0_norm;
    condF1_rec(1) = cond(F1_cached, inf);

    if m == 0
        J_chord = F1_cached;
    end

    iter = 0;
    converged = normF_rec(1) <= threshold;
    nonfinite = false;

    while ~converged && iter < iter_max
        Fx = F_cached;
        if m == 0
            J = J_chord;
        else
            J = F1_cached;
        end

        if any(~isfinite(Fx)) || any(~isfinite(J(:)))
            nonfinite = true;
            break;
        end

        delta = -J \ Fx;
        if any(~isfinite(delta))
            nonfinite = true;
            break;
        end

        x = x + delta;
        iter = iter + 1;

        F_cached  = F(x);
        F1_cached = F1(x);
        normF_rec(iter + 1)  = norm(F_cached, inf);
        condF1_rec(iter + 1) = cond(F1_cached, inf);

        if ~isfinite(normF_rec(iter + 1))
            nonfinite = true;
            break;
        end

        converged = normF_rec(iter + 1) <= threshold;
    end

    normF_rec  = normF_rec(1:iter + 1);
    condF1_rec = condF1_rec(1:iter + 1);

    if converged
        fprintf('Arresto: criterio sul residuo soddisfatto dopo %d iterazioni.\n', iter);
    elseif nonfinite
        fprintf('Arresto: valori non finiti incontrati dopo %d iterazioni.\n', iter);
    else
        fprintf('Arresto: raggiunto iter_max = %d senza soddisfare il criterio sul residuo.\n', iter_max);
    end
end


% Commento ai risultati:
% Il metodo di Newton aggiorna la Jacobiana a ogni passo e, quando
% l'innesco e' favorevole, converge molto rapidamente. Il metodo delle
% corde riusa la Jacobiana iniziale: costa meno per iterazione, ma puo'
% richiedere piu' iterazioni, arrestarsi per iter_max o divergere se la
% Jacobiana fissata non descrive bene il problema lungo il percorso. Nei
% sistemi nonlineari il punto iniziale influenza sensibilmente la
% convergenza.
% Da zero Newton converge, le corde possono divergere.
% Sul grafico semilogaritmico Newton mostra il tipico raddoppio asintotico
% delle cifre corrette per passo (convergenza quadratica vicino alla
% soluzione: ||F(x_{k+1})||_inf ~ C * ||F(x_k)||_inf^2), mentre le corde
% scendono geometricamente con rapporto legato a
% ||I - J(x_0)^{-1} J(x_*)||_inf.
% Il condizionamento di J_F lungo il percorso di Newton da (1,1,1,1) resta
% moderato e tendenzialmente decrescente avvicinandosi alla soluzione: i
% sistemi lineari interni non amplificano sensibilmente gli errori
% numerici, e l'errore relativo finale e' limitato essenzialmente dalla
% tolleranza sul residuo e dalla precisione di macchina.
