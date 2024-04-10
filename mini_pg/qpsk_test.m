format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[startpath,  filename, ext] = fileparts(full_path);
plot_num = 1;

commandwindow;

%%

sample_rate = 52e6;

num_bits = 384;

symbol_length = 272e-9;

channels = [-0.8:0.1:-0.1, 0.1:0.1:0.8]*10e6;
% channels = [0];

num_bursts = 16;

amplitude = 2000;

%% create the filters

% create the full filter using the window
fc = 1e6/sample_rate;
n_taps = 31;
w = nuttall_window(n_taps);

lpf = create_fir_filter(fc, w);

fc_h = 34e6/sample_rate;
lpf2 = create_fir_filter(fc_h, w);

apf = create_fir_filter(1.0, w);   %ones(n_taps,1)

hpf = 0.3*(apf-lpf2);
cpf = (lpf + hpf)/2.0;

fft_lpf = fft(lpf)/numel(lpf);
fft_lpf2 = fft(lpf2)/numel(lpf2);
fft_hpf = fft(hpf)/numel(hpf);

fft_apf = fft(apf)/numel(apf);
fft_cpf = fft(cpf)/numel(cpf);

% calculate the x axis
x_cpf = linspace(-sample_rate/2, sample_rate/2, numel(fft_cpf));

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(x_cpf/1e6, 20*log10(abs(fftshift(fft_lpf))),'b')
plot(x_cpf/1e6, 20*log10(abs(fftshift(fft_lpf2))),'g')
plot(x_cpf/1e6, 20*log10(abs(fftshift(fft_apf))),'r')
plot(x_cpf/1e6, 20*log10(abs(fftshift(fft_hpf))),'m')
plot(x_cpf/1e6, 20*log10(abs(fftshift(fft_cpf))),'k')

grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([x_cpf(1), x_cpf(end)]/1e6);
xlabel('Frequency (MHz)', 'fontweight', 'bold', 'FontSize', 13);
ylabel('Amplitude', 'fontweight', 'bold', 'FontSize', 13);
title('Frequency Response of LPF Filter', 'fontweight', 'bold', 'FontSize', 14);

plot_num = plot_num + 1;

%% loop through and create 16 bursts
iq_oqpsk = [];
for idx=1:num_bursts
    
    ch = channels(randi([1,numel(channels)],1,1));
    
    data = randi([0,1],num_bits,1);
    
    [iq] = generate_oqpsk(data, sample_rate, symbol_length/2);
    
    iq_cpf = conv(iq, lpf(end:-1:1), 'same');
    % iq_cpf = iq;
    
    iq_r = iq_cpf .* exp(2*1i*pi()*ch/sample_rate*(0:1:numel(iq_cpf)-1)).';
    
    iq_oqpsk = cat(1, iq_oqpsk, iq_r);
    
end

%%
figure(plot_num)
spectrogram(iq_oqpsk, 128, 64, 128, sample_rate, 'centered');
plot_num = plot_num + 1;


t_oqpsk = (0:1:numel(iq_oqpsk)-1)/sample_rate;
t_oqpsk(end)

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(t_oqpsk, real(iq_oqpsk),'b')
plot(t_oqpsk, imag(iq_oqpsk),'r')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
% xlim([x_bpsk(1), x_bpsk(end)]/1e6);
title('Plot of Filtered vs. Un-Filtered Samples', 'fontweight', 'bold', 'FontSize', 14);
plot_num = plot_num + 1;

%% plot the fft of the signals

fft_x0 = fft(iq_oqpsk(1:2695))/numel(iq_oqpsk(1:2695));
% fft_x1 = fft(x1)/numel(x1);

% calculate the x axis
x_oqpsk = linspace(-sample_rate/2, sample_rate/2, numel(fft_x0));

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(x_oqpsk/1e6, 20*log10(abs(fftshift(fft_x0))),'k')
% plot(x_oqpsk/1e6, 20*log10(abs(fftshift(fft_x1))),'g')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([x_oqpsk(1), x_oqpsk(end)]/1e6);
ylim([-100, 0]);
xlabel('Frequency (MHz)', 'fontweight', 'bold', 'FontSize', 13);
ylabel('Amplitude', 'fontweight', 'bold', 'FontSize', 13);
title('Filtered vs. Un-Filtered Signal', 'fontweight', 'bold', 'FontSize', 14);

plot_num = plot_num + 1;

return;

%% do the FFT on the signal
% fft_oqbpsk = fft(iq_oqpsk)/numel(iq_oqpsk);


figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(0:n_taps-1, lpf,'k')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([0, n_taps-1]);
title('Plot of Filter', 'fontweight', 'bold', 'FontSize', 14);

plot_num = plot_num + 1;

%% plot the results of the filter

% apply the filter to the bpsk signal
x1 = conv(iq_oqpsk, lpf(end:-1:1), 'same');

fft_lpf = fft(lpf)/numel(lpf);

% calculate the x axis
x_cpf = linspace(-sample_rate/2, sample_rate/2, numel(fft_lpf));

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(x_cpf/1e6, 20*log10(abs(fftshift(fft_lpf))),'k')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([x_cpf(1), x_cpf(end)]/1e6);
xlabel('Frequency (MHz)', 'fontweight', 'bold', 'FontSize', 13);
ylabel('Amplitude', 'fontweight', 'bold', 'FontSize', 13);
title('Frequency Response of LPF Filter', 'fontweight', 'bold', 'FontSize', 14);

plot_num = plot_num + 1;




%% create a band reject filter

fc = 28.6e6/sample_rate;
w = nuttall_window(n_taps);
lpf2 = create_fir_filter(fc, w);

apf = create_fir_filter(1.0, w);   %ones(n_taps,1)

hpf = 0.3*(apf-lpf2);

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(0:n_taps-1, hpf,'k')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([0, n_taps-1]);
title('Plot of Filter', 'fontweight', 'bold', 'FontSize', 14);

plot_num = plot_num + 1;

fft_hpf = fft(hpf)/numel(hpf);

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(x_cpf/1e6, 20*log10(abs(fftshift(fft_hpf))),'k')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([x_cpf(1), x_cpf(end)]/1e6);
xlabel('Frequency (MHz)', 'fontweight', 'bold', 'FontSize', 13);
ylabel('Amplitude', 'fontweight', 'bold', 'FontSize', 13);
title('Frequency Response of Band Reject Filter', 'fontweight', 'bold', 'FontSize', 14);
plot_num = plot_num + 1;

%% add filters together

cpf = (lpf + hpf)/2.0;

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(0:n_taps-1, cpf,'k')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([0, n_taps-1]);
title('Plot of Filter', 'fontweight', 'bold', 'FontSize', 14);

plot_num = plot_num + 1;

fft_cpf = fft(cpf)/numel(cpf);

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(x_cpf/1e6, 20*log10(abs(fftshift(fft_cpf))),'k')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([x_cpf(1), x_cpf(end)]/1e6);
xlabel('Frequency (MHz)', 'fontweight', 'bold', 'FontSize', 13);
ylabel('Amplitude', 'fontweight', 'bold', 'FontSize', 13);
title('Frequency Response of Band Reject Filter', 'fontweight', 'bold', 'FontSize', 14);
plot_num = plot_num + 1;

%% apply custom pass filter

x1_cpf = conv(1.5*iq_oqpsk, cpf(end:-1:1), 'same');

fft_x1_cpf = fft(x1_cpf)/numel(x1_cpf);

figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
hold on;
plot(x_oqpsk/1e6, 20*log10(abs(fftshift(fft_x0))),'k')
plot(x_oqpsk/1e6, 20*log10(abs(fftshift(fft_x1_cpf))),'g')
grid on
box on
set(gca,'fontweight','bold','FontSize', 13);
xlim([x_oqpsk(1), x_oqpsk(end)]/1e6);
ylim([-100,0]);
xlabel('Frequency (MHz)', 'fontweight', 'bold', 'FontSize', 13);
ylabel('Amplitude', 'fontweight', 'bold', 'FontSize', 13);
title('Band Reject Filter Result', 'fontweight', 'bold', 'FontSize', 14);
plot_num = plot_num + 1;

figure(plot_num)
spectrogram(x1_cpf, 128, 64, 128, sample_rate, 'centered');
plot_num = plot_num + 1;
