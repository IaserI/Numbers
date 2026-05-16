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

results = struct();

fprintf('Soluzione di riferimento fsolve:\n');
fprintf('  %.12e %.12e %.12e %.12e\n\n', x_star);

for idx_x0 = 1:size(x0_list, 2)
    x0 = x0_list(:, idx_x0);

    figure;
    hold on;

    for idx_method = 1:length(method_m)
        m_method = method_m(idx_method);
        [x, iter, normF_rec, condF1_rec, x_rec] = ...
            newtons(x0, F, F1, iter_max, tol, m_method);

        rel_err_rec = zeros(size(x_rec, 2), 1);
        for k = 1:size(x_rec, 2)
            rel_err_rec(k) = norm(x_rec(:, k) - x_star, inf) / norm(x_star, inf);
        end

        final_rel_err = norm(x - x_star, inf) / norm(x_star, inf);

        fprintf('Innesco %d, metodo %s: iter = %d, errore relativo finale = %.10e\n', ...
            idx_x0, method_names{idx_method}, iter, final_rel_err);

        semilogy(0:iter, rel_err_rec, '-o', 'LineWidth', 1.2);

        results(idx_x0, idx_method).x = x;
        results(idx_x0, idx_method).iter = iter;
        results(idx_x0, idx_method).normF_rec = normF_rec;
        results(idx_x0, idx_method).condF1_rec = condF1_rec;
        results(idx_x0, idx_method).x_rec = x_rec;
        results(idx_x0, idx_method).rel_err_rec = rel_err_rec;
    end

    grid on;
    title(sprintf('Errore relativo da x0 = (%g,%g,%g,%g)^T', x0));
    xlabel('Iterazione');
    ylabel('errore relativo in norma infinito');
    legend(method_names, 'Location', 'best');
end

iter_newton_ones = results(2, 2).iter;
condF1_rec_newton_ones = results(2, 2).condF1_rec;

figure;
plot(0:iter_newton_ones, condF1_rec_newton_ones, '-o', 'LineWidth', 1.2);
title('Condizionamento della Jacobiana - Newton da x0 = (1,1,1,1)^T');
xlabel('Iterazione');
ylabel('cond_inf(J_F(x_k))');
grid on;
legend('Newton', 'Location', 'best');

function [x, iter, normF_rec, condF1_rec, x_rec] = newtons(x, F, F1, iter_max, tol, m)
    if iter_max <= 0 || iter_max ~= floor(iter_max)
        error('newtons:InvalidIterMax', 'iter_max deve essere un intero positivo.');
    end

    if length(tol) ~= 2 || any(tol <= 0)
        error('newtons:InvalidTolerance', 'tol deve contenere due tolleranze positive.');
    end

    x = x(:);
    tau_a = tol(1);
    tau_r = tol(2);
    x_initial = x;
    F_initial = F(x_initial);
    F0_norm = norm(F_initial, inf);
    threshold = tau_a + tau_r*F0_norm;

    normF_rec = zeros(iter_max + 1, 1);
    condF1_rec = zeros(iter_max + 1, 1);
    x_rec = zeros(length(x), iter_max + 1);

    normF_rec(1) = F0_norm;
    condF1_rec(1) = cond(F1(x), inf);
    x_rec(:, 1) = x;

    if m == 0
        J_chord = F1(x_initial);
    end

    iter = 0;
    converged = normF_rec(1) <= threshold;
    nonfinite = false;

    while ~converged && iter < iter_max
        Fx = F(x);
        if m == 0
            J = J_chord;
        else
            J = F1(x);
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

        normF_rec(iter + 1) = norm(F(x), inf);
        condF1_rec(iter + 1) = cond(F1(x), inf);
        x_rec(:, iter + 1) = x;

        if ~isfinite(normF_rec(iter + 1))
            nonfinite = true;
            break;
        end

        converged = normF_rec(iter + 1) <= threshold;
    end

    normF_rec = normF_rec(1:iter + 1);
    condF1_rec = condF1_rec(1:iter + 1);
    x_rec = x_rec(:, 1:iter + 1);

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
% l'innesco e' favorevole, puo' convergere molto rapidamente. Il metodo
% delle corde riusa la Jacobiana iniziale: costa meno per iterazione, ma
% puo' richiedere piu' iterazioni, arrestarsi per iter_max o divergere se la
% Jacobiana fissata non descrive bene il problema lungo il percorso. Nei sistemi
% nonlineari il punto iniziale influenza sensibilmente la convergenza.
% L'andamento dell'errore relativo evidenzia la rapidita' di Newton nei casi
% favorevoli, mentre il condizionamento della Jacobiana indica quanto i
% sistemi lineari interni possono amplificare errori numerici.
