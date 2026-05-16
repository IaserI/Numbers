% Gruppo N
% Giovanni Bozzi, Tommaso Manca

clear;
clc;

n = 7;
m = 30;
s = linspace(0, 4, m)';
p = [cos(pi*s)./(s + 1), sin(pi*s)./(s + 1)];

[c0, t0] = fit_parametric_poly(0, n, p);
[c1, t1] = fit_parametric_poly(1, n, p);





%t_plot = linspace(0, 1, 1000)';
%V_plot = zeros(length(t_plot), n + 1);
%for j = 0:n
%    V_plot(:, j + 1) = t_plot.^j;
%end
%
%curve0 = V_plot*c0;
%curve1 = V_plot*c1;

%bombate
t_plot = linspace(0, 1, 1000)';
curva0 = [polyval(flip(c0(:,1)), t_plot), polyval(flip(c0(:,2)), t_plot)];
curva1 = [polyval(flip(c1(:,1)), t_plot), polyval(flip(c1(:,2)), t_plot)];



figure;
plot(p(:, 1), p(:, 2), 'ko', 'MarkerFaceColor', 'k');
hold on;
plot(curva0(:, 1), curva0(:, 2), 'b-', 'LineWidth', 1.5);
plot(p(:, 1), p(:, 2), 'k:', 'LineWidth', 0.8);
axis equal;
grid on;
title('Approssimazione parametrica, itype = 0');
xlabel('x');
ylabel('y');
legend('punti dati', 'curva grado 7', 'poligonale dati', 'Location', 'best');

figure;
plot(p(:, 1), p(:, 2), 'ko', 'MarkerFaceColor', 'k');
hold on;
plot(curva1(:, 1), curva1(:, 2), 'r-', 'LineWidth', 1.5);
plot(p(:, 1), p(:, 2), 'k:', 'LineWidth', 0.8);
axis equal;
grid on;
title('Approssimazione parametrica, itype = 1');
xlabel('x');
ylabel('y');
legend('punti dati', 'curva grado 7', 'poligonale dati', 'Location', 'best');

fprintf('Parametri itype=0 da %.2f a %.2f, itype=1 da %.2f a %.2f\n', ...
    t0(1), t0(end), t1(1), t1(end));


function [c, t] = fit_parametric_poly(itype, n, p)
    assert(isscalar(itype) && ismember(itype, [0,1]), 'itype deve essere 0 o 1');
    assert(isscalar(n) && n > 0 && mod(n,1) == 0, 'n deve essere intero positivo');
    assert(isnumeric(p) && ismatrix(p) && size(p,2) == 2 && ~isempty(p), 'Matrice p non valida ');

    m = size(p, 1);
    assert(m > n, 'm deve essere maggiore di n ');

    %parametrizzazione
    if itype == 0
        t = linspace(0, 1, m)';
    else
        segment_lengths = sqrt(sum(diff(p, 1, 1).^2, 2));
        Ltot = sum(segment_lengths);
        assert(Ltot > 0, 'La lunghezza totale della poligonale deve essere positiva');
        t = [0; cumsum(segment_lengths) / Ltot];
    end


    V = zeros(m, n + 1);
    for j = 0:n
        V(:, j + 1) = t.^j;
    end

    N = V'*V;
    rhs_x = V'*p(:, 1);
    rhs_y = V'*p(:, 2);

    cx = N \ rhs_x;
    cy = N \ rhs_y;
    c = [cx, cy];
end

% Commento ai risultati:
% Con itype = 0 i parametri sono equispaziati e quindi dipendono solo
% dall'indice dei punti, ignorando le distanze geometriche reali fra punti
% consecutivi. Con itype = 1 si usa invece la lunghezza di corda
% normalizzata, che distribuisce il parametro in modo piu' coerente con la
% geometria della poligonale. Le differenze fra le due curve vanno quindi
% lette rispetto alla disposizione dei punti nel piano. Il grado 7 puo'
% introdurre oscillazioni, ma il problema e' ai minimi quadrati e non
% richiede interpolazione esatta dei punti assegnati.
