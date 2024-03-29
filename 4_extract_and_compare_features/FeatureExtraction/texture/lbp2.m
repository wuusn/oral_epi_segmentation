function Feature_Vector = lbp(I)
warning off;


% Inline function accomplishes the following:
% 
%   Original          Thresholded      8-bit Conversion        Multiply 
% +-----------+      +-----------+      +-----------+       +-----------+
% | 5 | 4 | 3 |      | 1 | 1 | 1 |      | 1 | 2 | 4 |       | 1 | 2 | 4 |
% +---+---+---+      +---+---+---+      +---+---+---+       +---+---+---+
% | 4 | 3 | 1 | ---> | 1 |   | 0 | ---> | 8 |   |16 | --->  | 8 |   | 0 |
% +---+---+---+      +---+---+---+      +---+---+---+       +---+---+---+
% | 2 | 0 | 3 |      | 0 | 0 | 1 |      |32 |64 |128|       | 0 | 0 |128|
% +---+---+---+      +---+---+---+      +---+---+---+       +---+---+---+
%                                                 
% In this case, LBP=1+2+4+8+128=143.
% 
% The form of the 8-bit Conversion can be varied, as shown by the multiple
% examples:

lb = [128; 64; 32; 1; 0; 16; 2; 4; 8];
%   +-----------+
%   |128| 1 | 2 |
%   +---+---+---+
%   |64 |   | 4 |
%   +---+---+---+
%   |32 |16 | 8 |
%   +---+---+---+
%

lb = [2; 4; 8; 1; 0; 16; 128; 64; 32];
%   +-----------+
%   | 2 | 1 |128|
%   +---+---+---+
%   | 4 |   |64 |
%   +---+---+---+
%   | 8 |16 |32 |
%   +---+---+---+
%
% tic
lbpim = lbp_c(I,lb);
% toc


LUT         = zeros(1,256);
LUT(1)      = 1;
LUT(128)    = 1;
LUT(254)    = 1;
LUT(127)    = 1;
LUT(64)     = 2;
LUT(32)     = 2;
LUT(16)     = 2;
LUT(8)      = 2;
LUT(4)      = 2;
LUT(2)      = 2;
LUT(191)    = 2;
LUT(223)    = 2;
LUT(239)    = 2;
LUT(247)    = 2;
LUT(251)    = 2;
LUT(253)    = 2;
LUT(257)    = 1;
LUT(255)    = 1;
LUT(126)    = 2;
LUT(96)     = 2;
LUT(112)    = 2;
LUT(124)    = 2;
LUT(120)    = 2;
LUT(62)     = 2;
LUT(30)     = 2;
LUT(14)     = 2;
LUT(6)      = 2;
LUT(48)     = 2;
LUT(24)     = 2;
LUT(12)     = 2;
LUT(56)     = 2;
LUT(28)     = 2;
LUT(129)    = 2;
LUT(159)    = 2;
LUT(143)    = 2;
LUT(135)    = 2;
LUT(131)    = 2;
LUT(193)    = 2;
LUT(225)    = 2;
LUT(241)    = 2;
LUT(249)    = 2;
LUT(207)    = 2;
LUT(231)    = 2;
LUT(243)    = 2;
LUT(199)    = 2;
LUT(227)    = 2;

M = zeros(1,257);
for i = 1:numel(lbpim)
    if lbpim(i) == 0
        M(257) = M(257)+1;
    elseif LUT(lbpim(i))> 0
        M(lbpim(i)) = M(lbpim(i)) + 1;
    else
        M(256) = M(256) + 1;
    end
end

Feature_Vector = [M(257), M(1),M(2),M(4),M(6),M(8),M(12),M(14),M(16),M(24),M(28),M(30),M(32),M(64),M(48),M(56),M(62),M(96),M(112),M(120),M(124),M(126),M(127),M(128),M(129),M(131),M(135),M(143),M(159),M(191),M(193),M(199),M(207),M(223),M(225),M(227),M(231),M(239),M(241),M(243),M(247),M(249),M(251),M(253),M(254),M(255),M(256)];

warning on;
end