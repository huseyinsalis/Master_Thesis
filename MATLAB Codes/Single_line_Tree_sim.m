open_system('Desktop\\Thesis\Single_line_tree_impedance.slx');
tic
BlockPaths = find_system('Single_line_tree_impedance','Type','Block');
close all

blocklength = length(BlockPaths);

f = 100000;

% New Cable Parameters per foot
R_perfoot = 3.30E-03;
L_perfoot = 2.4E-10;
C_perfoot = 4.59E-15;
G_perfoot = 3.81E-18;

% R_perfoot = 6.86E-06;
% L_perfoot = 2.4E-10;
% C_perfoot = 4.59E-12;
% G_perfoot = 3.81E-18;
Data = [];
Lengths = [200 800; 400 600; 600 400; 800 200; 2000 8000; 4000 6000; 6000 4000; 8000 2000];
Tree_impedances = [1000, 10000, 25000, 50000, 75000, 100000];
GridRs = [];
GridLs = [];
GridRp = [];
GridCp = [];

%Generate block parameters
for i =1 :length(Lengths)
    [GridRs(i,1), GridLs(i,1), GridRp(i,1), GridCp(i,1)]= Calculate_Cable_Parameters(f, Lengths(i,1), R_perfoot, L_perfoot, C_perfoot, G_perfoot);
    [GridRs(i,2), GridLs(i,2), GridRp(i,2), GridCp(i,2)]= Calculate_Cable_Parameters(f, Lengths(i,2), R_perfoot, L_perfoot, C_perfoot, G_perfoot);
end


for i =1 : length(Lengths)
    %Assign new cable parameters to all blocks
    for j=1 : 2
    shuntr=strcat('Shunt_R',int2str(j));
    shuntc=strcat('Shunt_C',int2str(j));
    seriesr=strcat('Series_R',int2str(j));
    seriesl=strcat('Series_L',int2str(j));
    assignin('base',shuntr,GridRp(i,j));
    assignin('base',shuntc,GridCp(i,j));
    assignin('base',seriesr,GridRs(i,j));
    assignin('base',seriesl,GridLs(i,j));
    end
    for k =1 : length(Tree_impedances)
        Tree_Impedance= Tree_impedances(k);
        nameamp=strcat('amp','0');
        assignin('base',nameamp,1);
        sim('Single_line_tree_impedance');
        %Read Impedance and Phase values of each Node
        Name1=strcat('V','0');
        varname1=genvarname(Name1);
        Name2=strcat('I','0');
        varname2=genvarname(Name2);
        [Imp,Pha]=ImpPhase(eval(varname1),eval(varname2));
        
        Data=[Data; [Lengths(i,1), Tree_Impedance, Imp, Pha]];
    end    
end
        
function [Rs, Ls, Rp, Cp] = Calculate_Cable_Parameters(freq, l, Rper, Lper, Cper, Gper)
    w = 2 * pi * freq;
    series_impedance = complex(Rper, w*Lper);
    z = abs(series_impedance);

    parallel_admittance = complex(Gper*2, w*Cper*2);
    y = abs(parallel_admittance);
    gamma = sqrt(y*z);
    
    Z_prime = series_impedance*l*sinh(gamma*l)/(gamma*l);
    Y_prime = parallel_admittance*l*tanh(gamma*l/2)/(gamma*l/2);
   
    Rs = real(Z_prime);
    Ls = imag(Z_prime)/w;
    Rp = 1/ real(Y_prime/2);
    Cp = (imag(Y_prime/2)/w);

end

%Function to measure Impedance and Phase
function [Impedance, Phase] = ImpPhase(Voltage,Current)
    Vfft=fft(Voltage.Data);
    Ifft=fft(Current.Data);
    [mag_V, idx_V] = max(abs(Vfft));
    [mag_I, idx_I] = max(abs(Ifft));
    pV = angle(Vfft(idx_V));
    pI = angle(Ifft(idx_I));
    Phase = ((pV - pI)/pi)*180;
    Impedance = mag_V/mag_I;
end