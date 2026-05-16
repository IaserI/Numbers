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
report_name = 'report_esercizio3.txt';

fid = fopen(report_name, 'w');
if fid == -1
    error('esercizio3:ReportOpenError', 'Impossibile aprire il file di report.');
end

% scrittura su file
fprintf(fid, 'Gruppo N\nGiovanni Bozzi, Tommaso Manca\n\n');
fprintf(fid, 'Metodo SOR su matrici sparse random\n\n');
fprintf(fid, '%6s %8s %18s %12s\n', 'test', 'omega', 'errore_rel', 'iter');
fprintf(fid, '%s\n', repmat('-', 1, 50));

for test = 1:num_tests
    R = sprand(n, n, 1/n);
    R = R - spdiags(diag(R), 0, n, n);
    row_sums = sum(R, 2);
    A = R + spdiags(2 + row_sums, 0, n, n);

    % "soluzione esatta" calcolata una sola volta per le tre omega
    x_exact = A \ b;

    for k = 1:length(omegas)
        omega = omegas(k);
        [x, iter, ~] = sor_method(A, b, x0, omega, tol, nmax);

        rel_err = norm(x - x_exact, 2) / norm(x_exact, 2);
        errors(test, k) = rel_err;
        iterations(test, k) = iter;

        fprintf(fid, '%6d %8.2f %18.10e %12d\n', test, omega, rel_err, iter);
    end
end

fprintf(fid, '\nMedie aritmetiche sui %d esperimenti\n', num_tests);
fprintf(fid, '%8s %18s %18s\n', 'omega', 'media_errore', 'media_iter');
fprintf(fid, '%s\n', repmat('-', 1, 46));
for k = 1:length(omegas)
    fprintf(fid, '%8.2f %18.10e %18.10f\n', ...
        omegas(k), mean(errors(:, k)), mean(iterations(:, k)));
end

fclose(fid);

condA = cond(full(A));
figure;
spy(A);
title('Pattern ultima matrice A');
xlabel(sprintf('j   |   cond_2(A) = %.3e', condA));
ylabel('i');
grid on;

fprintf('Report in %s\n', report_name);




% function
function [x, iter, rel_res] = sor_method(A, b, x0, omega, tol, nmax)

    % controlli sugli input
    if ~ismatrix(A) || size(A, 1) ~= size(A, 2)
        error('sor_method:NonSquareInput', 'A deve essere una matrice quadrata.');
    end

    n = size(A, 1);
    if length(b) ~= n || length(x0) ~= n
        error('sor_method:InvalidDimensions', ...
            'Dimensioni non coerenti fra A, b e x0.');
    end

    if ~isscalar(omega) || omega <= 0
        error('sor_method:InvalidOmega', 'omega deve essere positivo.');
    end

    if ~isscalar(tol) || tol <= 0
        error('sor_method:InvalidTolerance', 'tol deve essere positiva.');
    end

    if ~isscalar(nmax) || nmax <= 0 || nmax ~= floor(nmax)
        error('sor_method:InvalidNmax', 'nmax deve essere un intero positivo.');
    end

    if any(abs(diag(A)) < eps)
        error('sor_method:ZeroDiagonal', ...
            'A ha elementi diagonali nulli o numericamente nulli.');
    end

    b = b(:);
    x = x0(:);
    norm_b = norm(b, 2);
    if norm_b == 0
        norm_b = 1;
    end

    % Splitting A = D + L_strict + U_strict precalcolato una sola volta:
    % la singola iterazione SOR equivale a
    %   (D + omega*L_strict) * (x_{k+1} - x_k) = omega * (b - A*x_k).
    % M e' sparsa triangolare bassa: "\" usa la forward substitution sparsa.
    M = omega * tril(A, -1) + spdiags(diag(A), 0, n, n);

    r = b - A*x;
    rel_res = norm(r, 2) / norm_b;
    iter = 0;

    while rel_res > tol && iter < nmax
        x = x + M \ (omega * r);
        iter = iter + 1;
        r = b - A*x;
        rel_res = norm(r, 2) / norm_b;
    end
end


% Commento ai risultati:
% Il criterio di arresto controlla il residuo relativo, non l'errore
% relativo rispetto alla soluzione esatta. Se A fosse molto mal
% condizionata, un residuo piccolo non garantirebbe un errore piccolo;
% qui le due quantita' sono dello stesso ordine perche' cond(A) e' moderato.
% Il valore omega = 1 coincide con Gauss-Seidel, omega < 1 e'
% sotto-rilassamento (piu' stabile ma piu' lento), omega > 1 e'
% sovra-rilassamento e puo' accelerare la convergenza se scelto bene: nei
% test omega = 1.1 e' tipicamente piu' veloce di omega = 1, che a sua volta
% lo e' piu' di omega = 0.5. Le differenze fra i 10 test dipendono dalle
% diverse realizzazioni random della matrice ma restano contenute.
% La matrice generata e' strettamente diagonale dominante per righe
% (A(i,i) = 2 + somma degli A(i,j) con j != i, off-diagonali non negativi),
% quindi SOR converge per ogni 0 < omega < 2 indipendentemente dalla
% particolare realizzazione random.
