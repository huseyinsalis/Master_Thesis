open_system('Desktop\\Thesis\Thesis_Grid_2.slx');
tic
BlockPaths = find_system('Thesis_Grid_2','Type','Block');
close all

blocklength = length(BlockPaths);

f = 100000;

% New Cable Parameters per foot
% New Cable Parameters per foot
R_perfoot = 3.3E-03;
L_perfoot = 1.2E-07;
C_perfoot = 2.1E-11;
G_perfoot = 3.5E-7;

rng(1234);
% randomnewCs = normrnd(2.1E-11,2E-13,[1 30]);
% randomnewGs = normrnd(3.5E-7,3E-8,[1 30]);
% Old Cable C Parameter per foot
C_perfoot_old = 2.1E-11*1.05;
G_perfoot_old = 3.5E-7*0.5;
% randomoldCs = normrnd(2.1E-11*1.05,2E-13,[1 50]);
% randomoldGs = normrnd(3.5E-7*0.5,3E-8,[1 50]);
% 
% figure(1)
% histogram(randomnewCs,10);
% xlabel('Capacitance (Farads)')
% hold on
% histogram(randomoldCs,10);
% legend('New','Old');
% figure(2)
% histogram(randomnewGs,10);
% xlabel('Conductance (Siemens)')
% hold on
% histogram(randomoldGs,10);
% legend('New','Old');

Data = cell(2500,35);
Lengths = [400, 175, 250, 200, 575, 500, 225, 650, 250, 250, 300, 250, 375, 350, 725, 800, 350, 550];
GridRs =[];
GridLs = [];
GridRp = [];
GridCp = [];
GridRs_old =[];
GridLs_old = [];
GridRp_old = [];
GridCp_old = [];

%Calculate each New cable parameters according to their length by using Long
%Transmission Line Model 11. cable will have old cable parameters
% for m=1 : 2
%     disp(m);
%     toc;
%     for n=1:2
       for i =1 :length(Lengths)
            disp(i);
            %if i == 1 && i==18 
             %  [GridRs(i), GridLs(i), GridRp(i), GridCp(i)]= Calculate_Cable_Parameters(f, Lengths(i), R_perfoot, L_perfoot, C_perfoot_old, G_perfoot_old);
             %   disp(randomoldCs(m));
             %   disp(GridCp(i));
            %else    
               [GridRs(i), GridLs(i), GridRp(i), GridCp(i)]= Calculate_Cable_Parameters(f, Lengths(i), R_perfoot, L_perfoot, C_perfoot, G_perfoot);
            %end
       end
       %Assign new cable parameters to all blocks
        for i =1 : length(Lengths)
            shuntr=strcat('Shunt_R',int2str(i));
            shuntc=strcat('Shunt_C',int2str(i));
            seriesr=strcat('Series_R',int2str(i));
            seriesl=strcat('Series_L',int2str(i));
            assignin('base',shuntr,GridRp(i));
            assignin('base',shuntc,GridCp(i));
            assignin('base',seriesr,GridRs(i));
            assignin('base',seriesl,GridLs(i));
        end
        
        %Assign all sources voltages to 0
        for i=0:17
            node=int2str(i);
            nameamp=strcat('amp',node);
            assignin('base',nameamp,0);
        end
        
        for i=0:17
            %Assign a source voltage to 1
            node=int2str(i);
            nameamp=strcat('amp',node);
            assignin('base',nameamp,1);
            %Assign previous sources voltage to 0
            if i>0 
                nodep=int2str(i-1);
                nameamp2=strcat('amp',nodep);
                assignin('base',nameamp2,0);
            end   
            % Run the simulation for each source
            sim('Thesis_Grid_2');
            % Read Impedance and Phase values of each Node
            Name1=strcat('V',node);
            varname1=genvarname(Name1);
            Name2=strcat('I',node);
            varname2=genvarname(Name2);
            [Imp,Pha]=ImpPhase(eval(varname1),eval(varname2));
            
            figure(2*i+1)
            plot (eval(varname1))
            title(Name1)
            figure(2*i+2)
            plot (eval(varname2))
            title(Name2)
            % Write each measurement to a table 
            Data{1,(2*i)+2}= Imp;
            Data{1,(2*i)+3}= Pha;
        end
        
%     end
% end 
% for k=0 : 17
%     figure(2*k+1)
%     plot (eval(genvarname(strcat('V',int2str(k)))))
%     title(strcat('V',int2str(k)))
%     figure(2*k+2)
%     plot (eval(genvarname(strcat('I',int2str(k)))))
%     title(strcat('I',int2str(k)))
% end
writetable(cell2table(Data), 'overhead.csv', 'writevariablenames', false)
%Function to Calculate cable paramters by using long transmission line
%model
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