%==========================================================================
% GEI-1090 - Detection of Voice - Programme d'acquisition des données
% F. Nougarou/Oumar Kone / Hiver2024
% %========================================================================
clear all;close all;clc;


Mots = 'droite';

Ind_M = [1];    % indice des mots
Nb_M =  50;     % Nombre d'essai par mots

Temps = 1;      % Temps de l'essai
deviceReader = audioDeviceReader;
deviceReader.SamplesPerFrame=1000;
deviceReader.SampleRate=44000;
setup(deviceReader)
Index_nom = 'Son_';
Mots_C = sprintf('%s%s',Index_nom,Mots);

%-- Chemin d'enregistrement -----------------------------------------------
chemin = sprintf('%s%s%s%s',pwd,'\sons_audio\',Mots_C,'\');
if ~ isfolder(chemin)
    mkdir(chemin)
end
%-- Ploting Config
figure;
MicPlot=plot(NaN,NaN,'XDataMode','auto','Color','r');
VoiceAmplitude_Buffer=zeros(Temps*deviceReader.SampleRate,1);

titre=sprintf('Voice Recoder ');
title(titre,'FontSize',12,...    %Titre du tracé
    'FontWeight','bold','FontName',...
    'Times New Roman','Color','k')

xlabel('Temps (s)','FontSize',12,... %Nom de l'axe des abscisses du tracé
    'FontWeight','bold','FontName',...
    'Times New Roman','Color','black')

ylabel('Magnitude','FontSize',12,... %Nom de l'axe des ordonnées du tracé
    'FontWeight','bold','FontName',...
    'Times New Roman','Color','black')
xlim([0 Temps*deviceReader.SampleRate]);
xticklabels({0:1/(10):Temps})
%          ylim([-1 ...
%              1])
grid on

set(gca,'NextPlot','replacechildren') ; % JUST Refresh the ploting data  not  the all thing ..idiot ^^

notvalidAudio = deviceReader(); %% to initialize it this data shall not be used
pause(0.1) % allow time to stabilize 
for ess = 1:Nb_M
    chemin_nom = sprintf('%sNF_N0%d_ES0%d.wav',chemin,Ind_M,ess);

    fileWriter = dsp.AudioFileWriter(chemin_nom,'FileFormat','WAV');
    if ~ strcmpi('Ambiance',Mots)
        fprintf('%d> Dites le mot %s\n',ess,Mots);
    end
   
    
    
    
    i=1;
    tic
   
   
    while i <= Temps*deviceReader.SampleRate/deviceReader.SamplesPerFrame
        acquiredAudio = deviceReader();

        %%% buffer for data plot
        VoiceAmplitude_Buffer= circshift(VoiceAmplitude_Buffer,1000);
        VoiceAmplitude_Buffer(i:i+999,:)=acquiredAudio;
        set(MicPlot,'YData',VoiceAmplitude_Buffer)
        drawnow limitrate
        fileWriter(acquiredAudio);
        % fprintf('Recording.... %.2f/%.2f (s)\n',toc,Temps);
        % fprintf('BatchNumber: %d/%d\n',i,deviceReader.SampleRate/deviceReader.SamplesPerFrame);

        i=i+1;
    end
   
    release(fileWriter)
    clc
    fprintf(2,'<strong> \n ==========>> Acquisition number : %d/%d\n </strong>',ess,Nb_M);
    fprintf('<strong>  Enregistrement complet \n </strong> ')  

end
release(deviceReader)
%-- Test de la lecture d'un mot
%  for ess = 1:Nb_M
%     chemin_nom = sprintf('%sNF_N0%d_ES0%d.wav',chemin,Ind_M,ess);
% pause(1)
% 
%
% [x,fs] = audioread(chemin_nom);
% sound(x,fs)
%  end
% 
%========================================================================


