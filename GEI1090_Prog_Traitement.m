%==========================================================================
% GEI-1090 - Detection of Voice - Programme de traitement
% F. Nougarou/Oumar Kone / Hiver2024
% =========================================================================
%
% Programme permet de faire :
% 1 - la lecture des données
% 2 - le pré-processing
% 3 - l'extraction des features
% 4 - la normation des features
% 4 - l'apprentissage des méthodes d'apprentissage automatique
% 5 - l'evaluation des méthodes
% 6 - le test en temps des méthodes
%
%==========================================================================
clear all;close all;clc;
fact = 4;
mode = 2;
mode_control = 1;

%==========================================================================
% 1 - Lecture des données
%==========================================================================

%-- Chemin des données audio ----------------------------------------------
chemin = sprintf('%s%s',pwd,'\sons_audio\');

%-- Création d'un magazin de données audio --------------------------------
ads = audioDatastore(chemin, 'IncludeSubfolders',true,...
    'FileExtensions','.wav', 'LabelSource','foldernames');
 
%-- Choix des sons à reconnaitre ------------------------------------------
commands = categorical(["Son_Haut","Son_Bas","Son_gauche","Son_droite","Son_ambiance"]);

NB_class = length(commands);

for n = 1:numel(commands)
    temp = char(commands(n));
    if strcmp(temp,'ambiance') == 0
        temp = temp(5:end-5);
    end
    temp_all{n} = temp;

end
commands_affiche = categorical(temp_all);
fs = 44e3;
Nb_ech_acq = fs;


isCommand = ismember(ads.Labels,commands); % Choix dans tous les sons
adsTrain = subset(ads,isCommand);          % Choix des dossisers du magasin
countEachLabel(adsTrain)                   % Compte des sons

%==========================================================================
% 2 - préparation au pré-processing
%==========================================================================

fs = 44e3;

%-- Filtrage --
frq = 3500/(fs/2);
type ="low";
ord =10;
[b,a] =butter(ord, frq, type);

%==========================================================================
% 3 - préparation à l'extraction de features
%==========================================================================

%-- Paramètre des features ------------------------------------------------
Temps_Frame = 0.03;
Saut_Temps = 0.02;
Taille_FFT = 512*4;
Temps_Segment = 1/fact;
Echantillon_Segment = round(Temps_Segment*fs);
Echantillon_Frame = round(Temps_Frame*fs);
Echantillon_Saut = round(Saut_Temps*fs);
Echantillon_Chevauche = Echantillon_Frame - Echantillon_Saut;

afe = audioFeatureExtractor( ...
    'SampleRate',fs, ...
    'FFTLength',Taille_FFT, ...
    'Window',hamming(Echantillon_Frame,'periodic'), ...
   'OverlapLength',Echantillon_Chevauche, ...   
    "mfcc",true);


%==========================================================================
% 4 - Boucle d'extraction des features
%==========================================================================


XTrain =[];
x_all = [];
iter = 0;
iter_ind = 0;
ind_all = [];

for ind = 1:numel(adsTrain.Files)
    x_all = [];
    %-- lecture des données --
    x = read(adsTrain);
    x_all = [x_all; x];
    
    %-- Filtrage --
    x_F = filter(b, a, x_all);
    %-- Feature extraction --
    features = extract(afe, x_F)';
    
   for n=1:fact
    iter=iter+1
    limit= ((Nb_ech_acq/fact)*(n-1)+1:(Nb_ech_acq/fact)*n);
   XTrain(:,:,iter) = extract(afe,x_F(limit,1))';
   end
    %XTrain(:,:,ind) = features;    
end


% --- Observation des features
XTrain_show = reshape(XTrain,size(XTrain,1),size(XTrain,2)*size(XTrain,3));
% figure(3);
% pcolor(XTrain_show);
% xlabel('Echantillons');
% ylabel('Frequences');
% title("Figure pcolor de XTrain_show");
% caxis([-4 2.6445]);
% shading flat



%--------------------------------
%-----questions II.B et II.C-----
%--------------------------------
%%-- Affichage Temporel et Fréquentiel --
% t = (0:length(x_all)-1) / fs; 
% figure(1);
% plot(t, x_all);
% xlabel('Temps (s)');
% ylabel('Magnitude');
% title('Signal dans le domaine temporel');
% grid on;
% 
% % Calculez la transformée de Fourier discrète (DFT) du signal x_all.
% N_x_all = length(x_all);
% X_ALL = fft(x_all);
% 
% % Calcul des fréquences associées à chaque composante de la DFT.
% f = (1:N_x_all)/N_x_all*fs;
% 
% X_ALL_magnitude = abs(X_ALL);
% % Tracez le spectre de fréquences.
% figure(2);
% plot(f, X_ALL_magnitude);
% xlabel('Fréquence (Hz)');
% ylabel('Magnitude');
% xlim([0 fs/2])
% title('Signal dans le domaine fréquentiel');
% grid on;
% 
% %déterminez les coefficients b et a d un filtre passe-bas
% frq = 3500/(fs/2);
% type ="low";
% ord =10;
% 
% [b,a] =butter(ord, frq, type);
% x_all_filtre = filter(b,a,x_all);
% sound(x_all_filtre,fs);
%==========================================================================
% 5 - Extraction des Labels
%==========================================================================
% --- Construction de xTrain_etal
XTrain_etal = reshape(XTrain, size(XTrain, 1) * size(XTrain, 2), size(XTrain, 3));

%==========================================================================
% 6 - Remise en forme des données avant passage dans les méthodes
%==========================================================================

%-- - Extraction des Labels

%-- Étallement des features --

%==========================================================================
% 6 - Normalisation
%==========================================================================

% %-- Calcul de la moyenne et de variance --
vect_mean = mean(XTrain_etal');
vect_var = var(XTrain_etal');
xTrain_Norm = (XTrain_etal-vect_mean')./vect_var';
%==========================================================================
% 7 - La construction du signal des Labels
%==========================================================================
YTrain = removecats(adsTrain.Labels);

%-- Construction et sauvegarde de la matrice pour le toolbox --
%--Convertir YTrain en un tableau de chaînes de caractères
YTrain = cellstr(YTrain);

%---- Récupérer les labels uniques
unique_labels = unique(YTrain); 
%---Créer une correspondance entre les labels et les valeurs numériques
target_mapping = containers.Map(unique_labels, 0:numel(unique_labels)-1); 
%-Initialiser le vecteur Target
Target =[];
for n=1:size(YTrain,1)
   [val,pos_ok]= max(YTrain(n,1)== commands);
   Target = [Target;pos_ok-1*ones(fact,1)];
end

%%%==================
% figure(5);
% subplot(2,1,1);
% pcolor(xTrain_Norm);
% xlabel('Echantillons');
% ylabel('Frequences');
% title("Figure pcolor des xTrain_Norm");
% caxis([-4 2.6445]);
% shading flat
% subplot(2,1,2);
% plot(Target);
% xlabel('Echantillons');
% ylabel('Frequences');
% title("Courbe du Target");

%==========================================================================
% 8 - L'entrainement et l'évaluation des méthodes d'apprentissage automatique
%==========================================================================

% %-- Construction et sauvegarde de la matrice pour le toolbox --
Data_ML = [xTrain_Norm; Target']';

%===============================================================================
%====la méthode Fine KNN et la méthode basée sur une réseau de neurone, notée RN
%===============================================================================

[trainedKNN_4, validationAccuracy] = KNN_4(Data_ML);
sons_estim = trainedKNN_4.predictFcn(Data_ML(:,1:end-1));

[m, order] = confusionmat(sons_estim, Target');
Diagonal = diag(m);sum_rows = sum(m,2); Precision = Diagonal./sum_rows;
Precision_KNN = mean(Precision)
%===================================================
%%%%%% SOUS COMMENTAIRES LA PARTIES III.B %%%%%%%%%%
%===================================================
% %----- Separation des donnees--------
% Data_ML_APP = [];
% Data_ML_DET = [];
% pourc_APP = 50;
% for n = 1:max(Data_ML(:,end))+1
%     pos_son = []; pos_son =find(Data_ML(:,end)== n-1);
%     Data_ML_APP = [Data_ML_APP; Data_ML(pos_son(1:floor(length(pos_son)*pourc_APP/100)),:)];
%     Data_ML_DET = [Data_ML_DET; Data_ML(pos_son(floor(length(pos_son)*pourc_APP/100)+1:end),:)];
% end
% 
% trainedKNN_1_APP = KNN_1(Data_ML_APP);
% % Re structuration Target pour 20% de DATA_ML
% Data_ML_APP_Target = Data_ML_APP(:,end);
% Data_ML_APP_Target_RN = zeros(4,size(Data_ML_APP_Target,1));
% for i = 1:4
%     start_col = (i - 1) * (size(Data_ML_APP_Target)/4) + 1;
%     end_col = i * (size(Data_ML_APP_Target)/4); 
%     % Attribuer 1 à la ligne correspondante et 0 aux autres lignes dans le groupe
%     Data_ML_APP_Target_RN(i, start_col:end_col) = (Data_ML_APP_Target(start_col:end_col) ~= 4);
%     start_col = start_col + (size(Data_ML_APP_Target)/4);
% end
% [net_APP, tr ] = train(net, Data_ML_APP(:,1:end-1)', Data_ML_APP_Target_RN);
% 
% sons_estim_KNN_APP = trainedKNN_1_APP.predictFcn(Data_ML_APP(:,1:end-1));
% y_sons_estim_RN_APP = net_APP(Data_ML_APP(:,1:end-1)');
% [val, pos] = max(y_sons_estim_RN_APP);
% sons_estim_RN_APP = pos -1;
% 
% sons_estim_KNN_DET = trainedKNN_1_APP.predictFcn(Data_ML_DET(:,1:end-1));
% y_sons_estim_RN_DET = net_APP(Data_ML_DET(:,1:end-1)');
% [val, pos] = max(y_sons_estim_RN_DET);
% sons_estim_RN_DET = pos -1;
% 
% Target_APP = Data_ML_APP(:,end);
% Target_DET = Data_ML_DET(:,end);
% 
% % Tracer la figure
% figure(7);
% 
% % Première sous-figure
% subplot(2,1,1);
% hold on;
% plot(Target_APP, 'b', 'LineWidth', 1.5);
% plot(sons_estim_KNN_APP, 'r--', 'LineWidth', 1.5);
% hold off;
% grid on;
% xlabel('Temps');
% ylabel('Classe');
% title('Comparaison des signaux Target\_APP et sons\_estim\_KNN\_APP');
% legend('Target\_APP', 'sons\_estim\_KNN\_APP');
% 
% % Deuxième sous-figure
% subplot(2,1,2);
% hold on;
% plot(Target_DET, 'b', 'LineWidth', 1.5);
% plot(sons_estim_KNN_DET, 'r--', 'LineWidth', 1.5);
% hold off;
% grid on;
% xlabel('Temps');
% ylabel('Classe');
% title('Comparaison des signaux Target_DET et sons\_estim\_KNN\_DET');
% legend('Target\_DET', 'sons\_estim\_KNN\_DET');
% 
% 
% % Tracer la figure
% figure(8);
% 
% % Première sous-figure
% subplot(2,1,1);
% hold on;
% plot(Target_APP, 'b', 'LineWidth', 1.5);
% plot(sons_estim_RN_APP, 'r--', 'LineWidth', 1.5);
% hold off;
% grid on;
% xlabel('Temps');
% ylabel('Classe');
% title('Comparaison des signaux Target\_APP et sons\_estim\_RN\_APP');
% legend('Target\_APP', 'sons\_estim\_RN\_APP');
% 
% % Seconde sous-figure
% subplot(2,1,2);
% hold on;
% plot(Target_DET, 'b', 'LineWidth', 1.5);
% plot(sons_estim_RN_DET, 'r--', 'LineWidth', 1.5);
% hold off;
% grid on;
% xlabel('Temps');
% ylabel('Classe');
% title('Comparaison des signaux Target_DET et sons\_estim\_RN\_DET');
% legend('Target\_DET', 'sons\_estim\_RN\_DET');
% 
% %==============================================
% %========CALCUL DE PRECISION ==================
% %==============================================
% [m, order] = confusionmat(sons_estim_KNN_APP, Target_APP');
% Diagonal = diag(m);sum_rows = sum(m,2); Precision = Diagonal./sum_rows;
% Precision_KNN_APP = mean(Precision);
% 
% [m, order] = confusionmat(sons_estim_KNN_DET, Target_DET');
% Diagonal = diag(m);sum_rows = sum(m,2); Precision = Diagonal./sum_rows;
% Precision_KNN_DET = mean(Precision);
% 
% [m, order] = confusionmat(sons_estim_RN_APP, Target_APP');
% Diagonal = diag(m);sum_rows = sum(m,2); Precision = Diagonal./sum_rows;
% Precision_RN_APP = mean(Precision);
% 
% [m, order] = confusionmat(sons_estim_RN_DET, Target_DET');
% Diagonal = diag(m);sum_rows = sum(m,2); Precision = Diagonal./sum_rows;
% Precision_RN_DET = mean(Precision);


%==========================================================================
% 9 - Le test en temps-réel des méthodes d'apprentissage automatique
%==========================================================================

%-- Mode d'affichage ------------------------------------------------------
% mode =2;  % si mode = 1 -> affichage estimation sinon --> Bubble Shooter

%-- Paramètres de lecture des données issues du microphone ----------------
fs = 44e3; SamplesPerFrame = Nb_ech_acq/fact;
adr = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',SamplesPerFrame);

%-- Paramètres pour l'affichage des données -------------------------------
close all
h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
if mode == 1
    subplot(2,1,1)
    title('Features')
    set(gca,'NextPlot','replacechildren')
    subplot(2,1,2)
    title('Sons estimés')
    set(gca, 'YTick', [0:numel(commands_affiche)], ...
        'YTickLabel',commands_affiche())
    ylim([0 numel(commands_affiche)-0.5])
    grid on;
    set(gca,'NextPlot','replacechildren')
end
val_touch = double(get(h,'CurrentCharacter'));

%%---- Initialisation pour le Bubble Shooter -----------------------------
MA0 = 0; time = 0; deplacement = []; switch_cercle = 1;
nb_test = 0; ech = 60; rst = 0; donne_affiche = 1;
curseurx = 0; curseury = 0; inside = 0;
couleur_cercle_cible = 'red'; couleur_cercle = 'blue';
vitesse1 = 0.4; vitesse2 = 0.4;
cunit = 4; nb_cercle = 16; pas = 360/nb_cercle;

angle = [-180+pas:pas:180];      %cercle extérieur
angle2 = [-180+pas*2:pas*2:180]; %cercle intérieur
a3 = randperm(numel(angle)); b3 = angle(a3(1));
xcunit = 4*cosd(b3); ycunit = 4*sind(b3);
r = 0.5; th = 0:pi/50:2*pi;
xunit = r * cos(th);
yunit = r * sin(th);

%-- Autres paramètres
timeLimit = 60; iter = 0; val_x = 0; val_y = 0; estim = []; affiche = 50;
tic;
val_estim = 0;
%-- Boucle de détection des sons en temps-réel ----------------------------
while  toc < timeLimit
    iter = iter +1;
    %-- Lecture des données -----------------------------------------------
    son_mic = adr();

    %-- Filtrage des données ----------------------------------------------
    son_mic = filter(b,a,son_mic);

    %-- Extraction des features -------------------------------------------
    features = extract(afe,son_mic)';

    %-- Etalement des données ---------------------------------------------
    features_etal = reshape(features,size(features,1)*size(features,2),1);

    %-- Normalisation -----------------------------------------------------
    features_etal_norm = (features_etal-vect_mean')./vect_var';

    %======================================================================
    %-- Prédiction des sons -----------------------------------------------
    %======================================================================
    if fact == 4
        estim(iter,1) = trainedKNN_4.predictFcn(features_etal_norm(1:end,1)');
    elseif fact == 2
        %-- Ajouter d'autre valeur de fact pour accélérer la vitesse

    end
    %-- Nom de son estimé et affichage ------------------------------------
    Nom_son_estim = commands(estim(iter,1)+1);
    message = sprintf('--> Nom : %s / Code : %d',Nom_son_estim,estim(iter,1));
    disp(message)

    %-- Affichage ---------------------------------------------------------
    val_touch = double(get(h,'CurrentCharacter'));
    if mode == 1
        %-- figure estimation --
        subplot(2,1,1)
        pcolor(features')
        shading flat
        caxis([-4 2.6445])
        subplot(2,1,2)
        if iter <= affiche
            plot(estim)
        else
            plot(estim(end-(affiche):end,1))
        end
        drawnow
        if val_touch == 27
            break
        end
    else
        %== BubbleShooter =================================================

        %-- Section 1 - définition des cercles ----------------------------
        % (VOUS N'AVEZ PAS TOUCHER CETTE SECTION)
        if MA0 == 0
            xcunit = 0;ycunit = 0;
            pos = [xcunit ycunit];
        else
            %pour définir le target du cercle aléatoire
            if switch_cercle == 1
                R = round(rand); a3 = []; b3 = [];
                if R == 0
                    a3 = randperm(numel(angle)); b3 = angle(a3(1));
                    xcunit = 4*cosd(b3); ycunit = 4*sind(b3);
                    pos = [pos; xcunit ycunit];
                else
                    a3 = randperm(numel(angle2)); b3 = angle2(a3(1));
                    xcunit = 2*cosd(b3); ycunit = 2*sind(b3);
                    pos = [pos; xcunit ycunit];
                end
                switch_cercle = 0;
            end
        end
        inside = inpolygon(curseurx,curseury,xunit+xcunit,yunit+ycunit);

        %==================================================================
        %-- Section 2 - contrôle du curseur -------------------------------
        %==================================================================
        
        %== vous allez jouer dans cette section ===========================
        stepSize=0.8;
        %== vous allez jouer dans cette section ===========================

        switch Nom_son_estim
            case 'Son_Haut'
                curseury = curseury + stepSize;
            case 'Son_Bas'
                curseury = curseury - stepSize;
            case 'Son_gauche'
                curseurx = curseurx - stepSize;
            case 'Son_droite'
                curseurx = curseurx + stepSize;
               
        end

        %-- Section 3 - Affichage -----------------------------------------
        % (VOUS N'AVEZ PAS TOUCHER CETTE SECTION)
        val_touch = []; val_touch = double(get(gcf,'CurrentCharacter'));
        %-- Affichage du curseur --
        if abs(curseurx) > 5;curseurx = sign(curseurx)*5;end
        if abs(curseury) > 5;curseury = sign(curseury)*5;end
        plot(curseurx, curseury,'color','blue','marker','+'); hold on;
        %-- Affichage des cercles --
        plot(xunit+xcunit, yunit+ycunit,couleur_cercle_cible); hold on;
        set(gca,'YTick',[],'XTick',[])
        xlim([-5 5]);ylim([-5 5])
        drawnow
        %-- Gestion des Bubble --------------------------------------------
        if rst >=donne_affiche;hold off;rst = 0;else;rst = rst +1;end
        if inside == 1;switch_cercle = 1;MA0 = 1;nb_test = nb_test+1;end
        if val_touch == 27;break;end
    end
end
%==========================================================================

