% Baseband PAM Signal Generation
% by Prof. Brian L. Evans
% The University of Texas at Austin
% Spring 2018
%
% m is sample index
% n is symbol index

format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[startpath,  filename, ext] = fileparts(full_path);
plot_num = 1;
line_width = 1;

commandwindow;

% Simulation parameters
N = 25;     % Number symbol periods
 
% Pulse shape g[m]
Ng = 4;     % Number symbol periods
L = 2000;     % Samples/symbol period
f0 = 1/L;
midpt = Ng*L/2;
m = (-midpt) : (midpt-1);
g = sinc(f0*m);

% Adjust for group delay
N = N + (Ng/2);
 
% M-level PAM symbol amplitudes
d = 3;
M = 4;
ioffset = (M + 1);
symAmp = (2*randi(M,[1,N]) - ioffset)*d;

% Discrete-time baseband PAM signal
mmax = N*L;
v = zeros(1,mmax);
v(1:L:end) = symAmp;  % interpolation
s = conv(v, g);       % pulse shaping
slength = length(s);  % trim result
s = s(midpt+1:slength-midpt+1);
 
% Interpretation in continuous time
Tsym = 0.001;       % Symbol period in sec
fsym = 1/Tsym;  % Symbol rate in Hz
fs = L*fsym;    % Sampling rate in Hz
Ts = 1/fs;      % Sampling time in sec

% Plots
Mmax = length(s);
m = 0 : (Mmax-1);
t = m*Ts;
Nmax = Mmax / L;
n = 0 : (Nmax-1);

figure;
plot(t,s);
hold on;
stem(n*Tsym,symAmp);
hold off;
xlim( [0 (Nmax-(Ng/2))*Tsym-Ts] );
ymax = Ng * (M-1) * d / 2;
ylim( [-ymax ymax] );
xlabel('Time (ms)');
title('Baseband PAM Signal s(t)');

%%
symAmp2 = upsample(symAmp, 200000);
Tsym2 = upsample(n*Tsym*10000, 10000);
s2=resample(s, 10000,1);
f = exp(pi()*1j*(500/numel(s2))*(0:(numel(s2)-1)));
ts = s2.*f;
figure;
plot(real(ts),'b');
hold on; 
plot(imag(ts),'r')
plot(s2,'g')
stem(symAmp2)


