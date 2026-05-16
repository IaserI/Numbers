% Gruppo N
% Giovanni Bozzi, Tommaso Manca


clear;
clc;

m = 30;
n = 7;
s = linspace(0, 4, m)';
p = [cos(pi*s), sin(pi*s)] ./ (s + 1);

c0 = fit_parametric_poly(0, n, p);
c1 = fit_parametric_poly(1, n, p);

% una sola griglia di valutazione e una sola matrice di Vandermonde,
% riutilizzate per entrambe le curve.
t_plot = linspace(0, 1, 1000)';
V_plot = t_plot.^(0:n);
curva0 = V_plot * c0;
curva1 = V_plot * c1;


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




% function
function c = fit_parametric_poly(itype, n, p)

    % controlli sugli input
    if ~isscalar(itype) || (itype ~= 0 && itype ~= 1)
        error('fit_parametric_poly:InvalidItype', ...
            'itype deve essere 0 oppure 1.');
    end

    if ~isscalar(n) || n <= 0 || mod(n, 1) ~= 0
        error('fit_parametric_poly:InvalidDegree', ...
            'n deve essere un intero positivo.');
    end

    if ~isnumeric(p) || ~ismatrix(p) || size(p, 2) ~= 2 || isempty(p)
        error('fit_parametric_poly:InvalidPoints', ...
            'p deve essere una matrice m x 2 non vuota.');
    end

    m = size(p, 1);
    if m <= n
        error('fit_parametric_poly:TooFewPoints', ...
            'Servono m > n punti per il problema ai minimi quadrati.');
    end

    % parametrizzazione: t_1 = 0 e t_m = 1 in entrambi i casi
    if itype == 0
        t = linspace(0, 1, m)';
    else
        segment_lengths = sqrt(sum(diff(p, 1, 1).^2, 2));
        Ltot = sum(segment_lengths);
        if Ltot <= 0
            error('fit_parametric_poly:DegeneratePolyline', ...
                'La lunghezza totale della poligonale deve essere positiva.');
        end
        t = [0; cumsum(segment_lengths) / Ltot];
    end

    % matrice di Vandermonde monomiale (m x (n+1)) via implicit expansion
    V = t.^(0:n);

    % equazioni normali risolte una sola volta per entrambe le coordinate:
    % N viene fattorizzato da "\" una sola volta e riusato sui due RHS.
    N = V' * V;
    c = N \ (V' * p);
end


% Commento ai risultati:
% Con itype = 0 i parametri sono equispaziati e dipendono solo dall'indice
% dei punti, ignorando le distanze geometriche reali fra punti consecutivi.
% Con itype = 1 si usa invece la lunghezza di corda normalizzata, che
% distribuisce il parametro in modo coerente con la geometria della
% poligonale: dove i punti sono piu' fitti, i t_i si addensano e la curva
% riceve piu' gradi di liberta' locali, mentre nei tratti piu' radi i t_i
% si distanziano. Il grado 7 puo' introdurre oscillazioni, ma il problema
% e' ai minimi quadrati e non richiede interpolazione esatta dei punti.
% Nota numerica: le equazioni normali sulla base monomiale comportano
% cond(V'V) = cond(V)^2, quindi gia' per n = 7 il sistema risolto e'
% percettibilmente piu' mal condizionato di V; il testo richiede comunque
% questa strada e per i dati in esame il risultato resta significativo.
