% Gruppo N
% Giovanni Bozzi, Tommaso Manca


clear;
clc;

nValues = 5:13;
report_name = 'report_esercizio1.txt';
fid = fopen(report_name, 'w');
if fid == -1
    error('esercizio1:ReportOpenError', 'Impossibile aprire il file di report.');
end

% scrittura su file
fprintf(fid, 'Gruppo N\nGiovanni Bozzi, Tommaso Manca\n\n');
fprintf(fid, 'Fattorizzazione LDLT su matrici di Hilbert\n\n');
fprintf(fid, '%4s %8s %18s %18s %18s\n','n', 'flag SDP', 'errore relativo', 'residuo relativo', 'cond_2(A)');
fprintf(fid, '%s\n', repmat('-', 1, 74));



for n = nValues
    A = hilb(n);
    xVero = 2*ones(n, 1);
    b = A*xVero;

    [L, D, flagSDP] = ldlt_custom(A);

    y = L \ b;
    z = y ./ D;
    x = L' \ z;

    errRel = norm(x - xVero, 2) / norm(xVero, 2);
    resRel = norm(b - A*x, 2) / norm(b, 2);
    condA = cond(A, 2);

    fprintf(fid, '%4d %8d %18.10e %18.10e %18.10e\n', ...
        n, flagSDP, errRel, resRel, condA);
end

fclose(fid);
fprintf('Report in %s\n', report_name);







%function
function [L, D, flagSDP] = ldlt_custom(A)

    %controlli sulla matrice A
    if ~ismatrix(A) || size(A, 1) ~= size(A, 2)
        error('ldlt_custom:NonSquareInput', 'La matrice A deve essere quadrata.');
    end

    if ~issymmetric(A)
        error('ldlt_custom:NonSymmetricInput', 'La matrice A deve essere simmetrica.');
    end


    n = size(A, 1);
    L = eye(n);
    D = zeros(n,1);

    zero_pivot_tol = eps*max(1, norm(A, 'fro'));
    flag_tol = 100*zero_pivot_tol;

    for j = 1:n
        somma = 0;
        for k = 1:j-1
            somma = somma + L(j, k)^2 * D(k);
        end
        D(j) = A(j, j) - somma;

        if abs(D(j)) <= zero_pivot_tol
            error('ldlt_custom:ZeroPivot', ...
                'Pivot nullo o numericamente nullo nella fattorizzazione LDL^T senza pivoting ');
        end

        for i = j+1:n
            somma = 0;
            for k = 1:j-1
                somma = somma + L(i, k)*D(k)*L(j, k);
            end
            L(i, j) = (A(i, j) - somma) / D(j);
        end
    end

    %controllo se A è SDP o no
    flagSDP = double(all(D >= -flag_tol));
end


% Le matrici di Hilbert sono definite positive e per tutti gli n inseriti
% si ha come risultato 1 del flag.
% Al crescere di n diventano estremamente mal condizionate: 
% i pivot della fattorizzazione LDL^T diventano così piccoli
% da poter essere contaminati dagli errori di approssimazione. Per questo
% il flag sdp può diventare numericamente meno affidabile per n grandi.
% Il residuo relativo può rimanere piccolo anche quando l'errore
% relativo cresce, perché il numero di condizionamento amplifica gli errori
% sui dati e sulle operazioni numeriche.