% #const w = 2.
% #const h = 2.

pixel((2*I+DI, 2*J+DJ)) : DI=0..1, DJ=0..1:- I=0..h-1, J=0..w-1.

tile((0, 0), (0, 0)):- pixel((2*h-1, 2*w-1)).
tile((0, J/2), (0, J)):- J=1..2*w-1, pixel((2*h-1, J-1)).
tile((I/2, 0), (I, 0)):- I=1..2*h-1, pixel((I-1, 2*w-1)).
tile((I/2, J/2), (I, J)):- I=1..2*h-1, J=1..2*w-1, pixel((I-1, J-1)).
tile_n((I, J), N):- I=0..h-1, J=0..w-1, { tile((I, J), _) }=N.

pair((K, L), (K+DI, L+DJ)):- K=(X1+1)/2, L=(Y1+1)/2, tile_n((I, J), (2;4)),
  tile((I, J), (X1, Y1)), tile((I, J), (X2, Y2)), (X1, Y1)!=(X2, Y2), DI=X2-X1, DJ=Y2-Y1.

h_paired(I):- pair((I, 0), _;(I, w), _), pair((I, J), _) : J=1..w-1.
strip_h_UB(I):- h_paired(I), not pair((I, _), (I+1, _)).
strip_h_LB(I):- h_paired(I), not pair((I, _), (I-1, _)).

strip_hc(I;I+1):- pair((I, J1), (I+1, J1+1); (I+1, J1), (I, J1+1)),
  pair((I, J2), (I+1, J2); (I, J2+1), (I+1, J2+1)), not tile_n((I, J1), 4).
strip_h(R):- strip_h_LB(LB), strip_h_UB(UB), LB<=UB, R=LB..UB,
  { not h_paired(R); strip_hc(R) }=0.

v_paired(J):- pair((0, J), _;(h, J), _), pair((I, J), _) : I=1..h-1.
strip_v_UB(J):- v_paired(J), not pair((_, J), (_, J+1)).
strip_v_LB(J):- v_paired(J), not pair((_, J), (_, J-1)).

strip_vc(J;J+1):- pair((I1, J), (I1+1, J+1); (I1+1, J), (I1, J+1)),
  pair((I2, J), (I2, J+1); (I2+1, J), (I2+1, J+1)), not tile_n((I1, J), 4).
strip_v(R):- strip_v_LB(LB), strip_v_UB(UB), LB<=UB, R=LB..UB,
  { not v_paired(R); strip_vc(R) }=0.
  
% :- not strip_h(_), not strip_v(_).

%#maximize { 1, h(I) : strip_h(I) ; 1, v(J) : strip_v(J) }.

% #show pixel/1.
% #show strip_h/1.
% #show strip_v/1.