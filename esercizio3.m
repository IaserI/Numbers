% Gruppo N
% Giovanni Bozzi, Tommaso Manca

clear;
clc;

n = 1000;
tol = 1e-8;
nmax = 500;
omegas = [0.5, 1, 1.1];
num_tests = 10;
x0 = zeros(n, 1);
i = (1:n)';
b = 1 + cos(2*pi*i/n);

errors = zeros(num_tests, length(omegas));
iterations = zeros(num_tests, length(omegas));
residuals = zeros(num_tests, length(omegas));
report_name = 'report_esercizio3.txt';

fid = fopen(report_name, 'w');
if fid == -1
    error('esercizio3:ReportOpenError', 'Impossibile aprire il file di report.');
end

fprintf(fid, 'Gruppo N\n');
fprintf(fid, 'Giovanni Bozzi, Tommaso Manca\n\n');
fprintf(fid, 'Metodo SOR su matrici sparse random\n\n');
fprintf(fid, '%6s %8s %18s %12s %18s\n', ...
    'test', 'omega', 'errore_rel', 'iter', 'residuo_rel');
fprintf(fid, '%s\n', repmat('-', 1, 72));

for test = 1:num_tests
    R = sprand(n, n, 1/n);
    R = R - spdiags(diag(R), 0, n, n);
    row_sums = sum(R, 2);
    A = R + spdiags(2 + row_sums, 0, n, n);

    x_exact = A \ b;

    for k = 1:length(omegas)
        omega = omegas(k);
        [x, iter, rel_res] = sor_method(A, b, x0, omega, tol, nmax);

        rel_err = norm(x - x_exact, 2) / norm(x_exact, 2);
        errors(test, k) = rel_err;
        iterations(test, k) = iter;
        residuals(test, k) = rel_res;

        fprintf(fid, '%6d %8.2f %18.10e %12d %18.10e\n', ...
            test, omega, rel_err, iter, rel_res);
    end
end

fprintf(fid, '\nMedie aritmetiche sui %d esperimenti\n', num_tests);
fprintf(fid, '%8s %18s %18s %18s\n', ...
    'omega', 'media_errore', 'media_iter', 'media_residuo');
fprintf(fid, '%s\n', repmat('-', 1, 68));
for k = 1:length(omegas)
    fprintf(fid, '%8.2f %18.10e %18.10f %18.10e\n', ...
        omegas(k), mean(errors(:, k)), mean(iterations(:, k)), mean(residuals(:, k)));
end

fclose(fid);

condA = cond(full(A));
figure;
spy(A);
title('Pattern ultima matrice A');
xlabel(sprintf('j   |   cond_2(A) = %.3e', condA));
ylabel('i');
grid on;

fprintf('Report scritto in %s\n', report_name);

function [x, iter, rel_res, rel_res_history] = sor_method(A, b, x0, omega, tol, nmax)
    if ~ismatrix(A) || size(A, 1) ~= size(A, 2)
        error('sor_method:NonSquareInput', 'A deve essere una matrice quadrata.');
    end

    n = size(A, 1);
    if length(b) ~= n || length(x0) ~= n
        error('sor_method:InvalidDimensions', 'Dimensioni non coerenti fra A, b e x0.');
    end

    if omega <= 0
        error('sor_method:InvalidOmega', 'omega deve essere positivo.');
    end

    if tol <= 0
        error('sor_method:InvalidTolerance', 'tol deve essere positiva.');
    end

    if nmax <= 0 || nmax ~= floor(nmax)
        error('sor_method:InvalidNmax', 'nmax deve essere un intero positivo.');
    end

    if any(abs(diag(A)) < eps)
        error('sor_method:ZeroDiagonal', 'A ha elementi diagonali nulli o numericamente nulli.');
    end

    b = b(:);
    x = x0(:);
    norm_b = norm(b, 2);
    if norm_b == 0
        norm_b = 1;
    end

    rel_res = norm(b - A*x, 2) / norm_b;
    rel_res_history = zeros(nmax + 1, 1);
    rel_res_history(1) = rel_res;
    iter = 0;

    while rel_res > tol && iter < nmax
        iter = iter + 1;
        x_old = x;

        for row = 1:n
            sigma1 = A(row, 1:row-1) * x(1:row-1);
            sigma2 = A(row, row+1:n) * x_old(row+1:n);
            x(row) = (1 - omega)*x_old(row) + ...
                omega*(b(row) - sigma1 - sigma2) / A(row, row);
        end

        rel_res = norm(b - A*x, 2) / norm_b;
        rel_res_history(iter + 1) = rel_res;
    end

    rel_res_history = rel_res_history(1:iter + 1);
end

% Commento ai risultati:
% Il criterio di arresto del metodo SOR controlla il residuo relativo, non
% direttamente l'errore relativo rispetto alla soluzione esatta. Se la
% matrice fosse molto mal condizionata, un residuo piccolo non garantirebbe
% necessariamente un errore piccolo. In questi esperimenti la dominanza
% diagonale per righe favorisce la convergenza. Il valore omega = 1 coincide
% con Gauss-Seidel, omega < 1 e' sotto-rilassamento e puo' essere piu'
% stabile ma piu' lento, mentre omega > 1 e' sovra-rilassamento e puo'
% accelerare la convergenza se scelto bene. Le differenze fra i 10 test
% dipendono dalle diverse matrici sparse random generate.
