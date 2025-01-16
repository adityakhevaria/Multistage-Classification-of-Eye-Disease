% Main MATLAB script for eye disease detection
clc;
clear all;
close all;
close all hidden;
warning off;

% Load the input image
img = imread('input_image.png');
img = imresize(img, [224 224]);
figure, imshow(img);

% The rest of the MATLAB code goes here...
%% Step 2: Preprocess the image (Noise Removal using Median Filtering)
redChannel = img(:, :, 1);
greenChannel = img(:, :, 2);
blueChannel = img(:, :, 3);

filteredRed = medfilt2(redChannel);
filteredGreen = medfilt2(greenChannel);
filteredBlue = medfilt2(blueChannel);

filteredRGB = cat(3, filteredRed, filteredGreen, filteredBlue);
figure, imshow(filteredRGB);
title('Noise Removed Image');

%% Step 3: Contrast Enhancement (Restored Image)
restoredImage = imadjust(filteredRGB, [0.2, 0.8], [0, 1]);
figure;
imshow(restoredImage);
title('Restored Image');

%% Step 4: Image Processing
Segmented_Image = MFORG(restoredImage);
figure;
imshow(Segmented_Image);
title('Overlay of Segmented Image on Original Image');

%%

matlabroot = cd;    % Dataset path
datasetpath = fullfile(matlabroot,'DR - Disease Dataset');   %Build full file name from parts
imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

[imdsTrain, imdsValidation] = splitEachLabel(imds,0.8,'randomized');     %Split ImageDatastore labels by proportions

augimdsTrain = augmentedImageDatastore([224 224 3], imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 3], imdsValidation);

%%

net = densenet201

layers = [imageInputLayer([224 224 3])

  net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers

  fullyConnectedLayer(2) % modifying the fullyconnected layer with respect to classes

  softmaxLayer

  classificationLayer];


options = trainingOptions("sgdm",...
  "ExecutionEnvironment","auto",...
  "InitialLearnRate",0.001,...
  "MaxEpochs",100,...
  "MiniBatchSize",25,...
  "Shuffle","every-epoch",...
  "ValidationData",augimdsValidation, ...
  "ValidationFrequency",50,...
  "Verbose",true, ...
  "Plots","training-progress");


% [net, traininfo] = trainNetwork(augimdsTrain,layers,options);  %Train neural network for deep learning

load net
load traininfo

YPred = classify(net,Segmented_Image);
msgbox(char(YPred))

Accuracy = mean(traininfo.TrainingAccuracy);

% Mertics
YPredTest = classify(net, augimdsValidation); % Use classify for validation data
YTest = imdsValidation.Labels;

PSNR = psnr(img,Segmented_Image);
Entropy = entropy(img);
[confMat, order] = confusionmat(YTest, YPredTest);


% Calculate metrics
TP = diag(confMat);
FP = sum(confMat, 1)' - TP;
FN = sum(confMat, 2) - TP;
TN = sum(confMat(:)) - (TP + FP + FN);

specificityP = TN ./ (TN + FP);
sensitivityP = TP ./ (TP + FN);
precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
f1_score = 2 * (precision .* recall) ./ (precision + recall);
AUC = 0.5 * (TP ./ (TP + FN) + TN ./ (TN + FP));
AUC = mean(AUC);

Metrics.Accuracy = Accuracy;
Metrics.PSNR = PSNR;
Metrics.Entropy = Entropy;
Metrics.AUC = AUC* 100;
Metrics.Precision = mean(precision) * 100;
Metrics.Recall = mean(recall) * 100;
Metrics.F1_Score = mean(f1_score) * 100;
Metrics.Specificity = mean(specificityP) * 100;
Metrics.Sensitivity = mean(sensitivityP) * 100;
disp("Diabetic Retinopathy")
disp(Metrics);

%%
output = char(YPred);

if strcmp(output, 'Diabetic Retinopathy')
  %% Image Processing Techniques
  % Step 1: Morphological Closing Operation
  se = strel('disk', 10); % Create a disk-shaped structuring element with radius 10
  closeBW = imclose(img, se); % Perform morphological closing operation
  figure, imshow(closeBW);
  title('Morphologically Closed Image');

  % Step 2: Adaptive Contrast Enhancement using LAB color space
  shadow_lab = rgb2lab(closeBW); % Convert the closed image to LAB color space
  max_luminosity = 100;
  L = shadow_lab(:,:,1)/max_luminosity; % Extract the L (luminosity) channel

  shadow_imadjust = shadow_lab;
  shadow_imadjust(:,:,1) = imadjust(L) * max_luminosity; % Apply contrast enhancement on the L channel
  shadow_imadjust = lab2rgb(shadow_imadjust); % Convert back to RGB

  figure, imshow(shadow_imadjust);
  title('Contrast Enhanced Image');

  % Step 3: Gabor Filter Application
  I = rgb2gray(shadow_imadjust); % Convert enhanced image to grayscale
  wavelength = 4; % Set the wavelength for the Gabor filter
  orientation = 90; % Set the orientation for the Gabor filter
  [mag, phase] = imgaborfilt(I, wavelength, orientation); % Apply Gabor filter

  figure;
  tiledlayout(1,2);
  nexttile;
  imshow(mag, []); % Display the magnitude of the Gabor filter
  title('Gabor Magnitude');
  nexttile;
  imshow(phase, []); % Display the phase of the Gabor filter
  title('Gabor Phase');
  % Step 4: Thresholding using Otsu's Algorithm
  level = graythresh(phase); % Compute Otsu's threshold level
  BW = imbinarize(phase, level); % Binarize the phase image using the threshold
  figure, imshow(BW);
  title('Binary Map using Otsu Thresholding');
  % Step 5: Overlay Binary Map on Original Image
  colorImageOverlay = imoverlay(img, BW, [1 0 0]); % Overlay binary map on the original image in red
  figure, imshow(colorImageOverlay);
  title('Final Color Image with Binary Overlay');

  Diseasematlabroot = cd;    % Dataset path
  Diseasedatasetpath = fullfile(Diseasematlabroot,'DR - DiseaseStage Dataset');   %Build full file name from parts
  Diseaseimds = imageDatastore(Diseasedatasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

  [DiseaseimdsTrain, DiseaseimdsValidation] = splitEachLabel(Diseaseimds,0.8);     %Split ImageDatastore labels by proportions

  DiseaseaugimdsTrain = augmentedImageDatastore([224 224 3],DiseaseimdsTrain);  %Generate batches of augmented image data
  DiseaseaugimdsValidation = augmentedImageDatastore([224 224 3],DiseaseimdsValidation);
  % Training Options
  Diseaseoptions = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",100,...
    "MiniBatchSize",25,...
    "Shuffle","every-epoch",...
    "ValidationData",DiseaseaugimdsTrain, ...
    "ValidationFrequency",50,...
    "Verbose",true, ...
    "Plots","training-progress");

  Diseaselayers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

  %       [Diseasenet, Diseasetraininfo] = trainNetwork(DiseaseaugimdsTrain,Diseaselayers,Diseaseoptions);  %Train neural network for deep learning

  load Diseasenet
  load Diseasetraininfo

  [DiseaseYPred,Diseasescore] = classify(Diseasenet,colorImageOverlay);      %Classify data using a trained deep learning neural network
  msgbox(char(DiseaseYPred));


  YPredTest_Stage = classify(Diseasenet, DiseaseaugimdsValidation); % Use classify for validation data
  YTest = DiseaseimdsValidation.Labels;

  DiseaseAccuracy = mean(Diseasetraininfo.TrainingAccuracy);

  PSNR_Stage = psnr(img,colorImageOverlay );

  Entropy_Stage = entropy(img);


  [confMat_Stage, order] = confusionmat(YTest, YPredTest_Stage);

  % Calculate metrics
  TP_Stage = diag(confMat_Stage);
  FP_Stage = sum(confMat_Stage, 1)' - TP_Stage;
  FN_Stage = sum(confMat_Stage, 2) - TP_Stage;
  TN_Stage = sum(confMat_Stage(:)) - (TP_Stage + FP_Stage + FN_Stage);

  specificityP_Stage = TN_Stage ./ (TN_Stage + FP_Stage);
  sensitivityP_Stage = TP_Stage ./ (TP_Stage + FN_Stage);  % Sensitivity calculation
  precision_Stage = TP_Stage ./ (TP_Stage + FP_Stage);
  recall_Stage = TP_Stage ./ (TP_Stage + FN_Stage);
  f1_score_Stage = 2 * (precision_Stage .* recall_Stage) ./ (precision_Stage + recall_Stage);
  AUC_Stage = 0.5 * (TP_Stage ./ (TP_Stage + FN_Stage) + TN_Stage ./ (TN_Stage + FP_Stage));
  AUC_Stage = mean(AUC_Stage);

  StageMetrics.Accuracy = DiseaseAccuracy;
  StageMetrics.PSNR = PSNR_Stage;
  StageMetrics.Entropy = Entropy_Stage;
  StageMetrics.AUC = AUC_Stage* 100;
  StageMetrics.Precision = mean(precision_Stage) * 100;
  StageMetrics.Recall = mean(recall_Stage) * 100;
  StageMetrics.F1_Score = mean(f1_score_Stage) * 100;
  StageMetrics.Specificity = mean(specificityP_Stage) * 100;
  StageMetrics.Sensitivity = mean(sensitivityP_Stage) * 100;
  disp("Stages Diabetic Retinopathy")
  disp(StageMetrics);

  DiseaseYPred = char(DiseaseYPred);

  if strcmp(DiseaseYPred, 'Mild DR')
    TreatmentOptions = 'Strict control of blood sugar, blood pressure, and cholesterol. Regular eye exams to monitor progression.';
    LifestyleRecommendations = ' Maintain a healthy diet, exercise regularly, avoid smoking, and manage stress. Regular blood sugar monitoring is crucial.';
  elseif strcmp(DiseaseYPred, 'Moderate DR')
    TreatmentOptions = ' In addition to controlling blood sugar, anti-VEGF injections may be recommended. Laser treatments can help reduce the risk of vision loss.';
    LifestyleRecommendations = ' Continue to monitor blood sugar levels, and follow dietary guidelines. Exercise and avoid activities that strain the eyes.';
  elseif strcmp(DiseaseYPred, 'Proliferative DR')
    TreatmentOptions = ' Laser surgery (photocoagulation) to prevent new blood vessels from forming. Vitrectomy may be necessary for advanced cases.';
    LifestyleRecommendations = ' Maintain close contact with your healthcare provider. Adhere strictly to treatments, and avoid activities that might increase eye pressure.';
  elseif strcmp(DiseaseYPred, 'Severe DR')
    TreatmentOptions = ' Frequent anti-VEGF injections, vitrectomy surgery, or laser therapy. In some cases, advanced treatment options like retinal surgery may be required.';
    LifestyleRecommendations = ' Focus on managing diabetes and overall health. Engage in low-impact physical activity and attend regular eye and medical check-ups. Emotional and mental support is important.';
  else
  end
  generateDiagnosisDR_Report(img, Segmented_Image, colorImageOverlay, DiseaseYPred, TreatmentOptions, LifestyleRecommendations, Metrics, StageMetrics)

elseif strcmp(output, 'Healthy')
  DiseaseYPred = 'No Diabetic Retinopathy';
  TreatmentOptions = ': The patient is healthy, with no signs of diabetic retinopathy.';
  LifestyleRecommendations = ' Continue to manage blood sugar levels, follow a healthy diet, engage in regular physical activity, and have regular eye examinations.';
  generateDiagnosisDR_Reporthealthy(img, Segmented_Image, DiseaseYPred, TreatmentOptions, LifestyleRecommendations,Metrics)
end
%%
%%
%%
ME_matlabroot = cd;    % Dataset path
ME_datasetpath = fullfile(ME_matlabroot,'ME- Disease Dataset');   %Build full file name from parts
ME_imds = imageDatastore(ME_datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

[ME_imdsTrain, ME_imdsValidation] = splitEachLabel(ME_imds,0.8,'randomized');     %Split ImageDatastore labels by proportions

ME_augimdsTrain = augmentedImageDatastore([224 224 3], ME_imdsTrain);
ME_augimdsValidation = augmentedImageDatastore([224 224 3], ME_imdsValidation);


ME_net = densenet201

ME_layers = [imageInputLayer([224 224 3])

  ME_net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers

  fullyConnectedLayer(2) % modifying the fullyconnected layer with respect to classes

  softmaxLayer

  classificationLayer];


ME_options = trainingOptions("sgdm",...
  "ExecutionEnvironment","auto",...
  "InitialLearnRate",0.001,...
  "MaxEpochs",100,...
  "MiniBatchSize",25,...
  "Shuffle","every-epoch",...
  "ValidationData",ME_augimdsValidation, ...
  "ValidationFrequency",50,...
  "Verbose",true, ...
  "Plots","training-progress");


% [ME_net, ME_traininfo] = trainNetwork(ME_augimdsTrain,ME_layers,ME_options);  %Train neural network for deep learning

load ME_net
load ME_traininfo

ME_YPred = classify(ME_net,Segmented_Image);
msgbox(char(ME_YPred))

ME_Accuracy = mean(ME_traininfo.TrainingAccuracy);

% Mertics
ME_YPredTest = classify(ME_net, ME_augimdsValidation); % Use classify for validation data
ME_YTest = ME_imdsValidation.Labels;

ME_PSNR = psnr(img,Segmented_Image);
ME_Entropy = entropy(img);
[ME_confMat, ME_order] = confusionmat(ME_YTest, ME_YPredTest);

% Calculate metrics
ME_TP = diag(ME_confMat);
ME_FP = sum(ME_confMat, 1)' - ME_TP;
ME_FN = sum(ME_confMat, 2) - ME_TP;
ME_TN = sum(ME_confMat(:)) - (ME_TP + ME_FP + ME_FN);

ME_specificityP = ME_TN ./ (ME_TN + ME_FP);
ME_sensitivityP = ME_TP ./ (ME_TP + ME_FN);
ME_precision = ME_TP ./ (ME_TP + ME_FP);
ME_recall = ME_TP ./ (ME_TP + ME_FN);
ME_f1_score = 2 * (ME_precision .* ME_recall) ./ (ME_precision + ME_recall);
ME_AUC = 0.5 * (ME_TP ./ (ME_TP + ME_FN) + ME_TN ./ (ME_TN + ME_FP));
ME_AUC = mean(ME_AUC)* 100;

ME_Metrics.Accuracy = ME_Accuracy;
ME_Metrics.PSNR = ME_PSNR;
ME_Metrics.Entropy = ME_Entropy;
ME_Metrics.AUC = ME_AUC;
ME_Metrics.Precision = mean(ME_precision) * 100;
ME_Metrics.Recall = mean(ME_recall) * 100;
ME_Metrics.F1_Score = mean(ME_f1_score) * 100;
ME_Metrics.Specificity = mean(ME_specificityP) * 100;
ME_Metrics.Sensitivity = mean(ME_sensitivityP) * 100;
disp(" Macular Edema")
disp(ME_Metrics);


%%
ME_output = char(ME_YPred);

if strcmp(ME_output, 'Macular Edema')
  %% Image Processing Techniques
  % Step 1: Morphological Closing Operation
  ME_se = strel('disk', 10); % Create a disk-shaped structuring element with radius 10
  ME_closeBW = imclose(img, ME_se); % Perform morphological closing operation


  % Step 2: Adaptive Contrast Enhancement using LAB color space
  ME_shadow_lab = rgb2lab(ME_closeBW); % Convert the closed image to LAB color space
  ME_max_luminosity = 100;
  ME_L = ME_shadow_lab(:,:,1)/ME_max_luminosity; % Extract the L (luminosity) channel

  ME_shadow_imadjust = ME_shadow_lab;
  ME_shadow_imadjust(:,:,1) = imadjust(ME_L) * ME_max_luminosity; % Apply contrast enhancement on the L channel
  ME_shadow_imadjust = lab2rgb(ME_shadow_imadjust); % Convert back to RGB


  % Step 3: Gabor Filter Application
  ME_I = rgb2gray(ME_shadow_imadjust); % Convert enhanced image to grayscale
  ME_wavelength = 4; % Set the wavelength for the Gabor filter
  ME_orientation = 90; % Set the orientation for the Gabor filter
  [ME_mag, ME_phase] = imgaborfilt(ME_I, ME_wavelength, ME_orientation); % Apply Gabor filter

  % Step 4: Thresholding using Otsu's Algorithm
  ME_level = graythresh(ME_phase); % Compute Otsu's threshold level
  ME_BW = imbinarize(ME_phase, ME_level); % Binarize the phase image using the threshold

  % Step 5: Overlay Binary Map on Original Image
  ME_colorImageOverlay = imoverlay(img, ME_BW, [1 0 0]); % Overlay binary map on the original image in red


  ME_Diseasematlabroot = cd;    % Dataset path
  ME_Diseasedatasetpath = fullfile(ME_Diseasematlabroot,'ME- DiseaseStage Dataset');   %Build full file name from parts
  ME_Diseaseimds = imageDatastore(ME_Diseasedatasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

  [ME_DiseaseimdsTrain, ME_DiseaseimdsValidation] = splitEachLabel(ME_Diseaseimds,0.8);     %Split ImageDatastore labels by proportions

  ME_DiseaseaugimdsTrain = augmentedImageDatastore([224 224 3],ME_DiseaseimdsTrain);  %Generate batches of augmented image data
  ME_DiseaseaugimdsValidation = augmentedImageDatastore([224 224 3],ME_DiseaseimdsValidation);
  % Training Options
  ME_Diseaseoptions = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",100,...
    "MiniBatchSize",25,...
    "Shuffle","every-epoch",...
    "ValidationData",ME_DiseaseaugimdsTrain, ...
    "ValidationFrequency",50,...
    "Verbose",true, ...
    "Plots","training-progress");

  ME_Diseaselayers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

  %       [ME_Diseasenet, ME_Diseasetraininfo] = trainNetwork(ME_DiseaseaugimdsTrain,ME_Diseaselayers,ME_Diseaseoptions);  %Train neural network for deep learning

  load ME_Diseasenet
  load ME_Diseasetraininfo

  [ME_DiseaseYPred,ME_Diseasescore] = classify(ME_Diseasenet,ME_colorImageOverlay);      %Classify data using a trained deep learning neural network
  msgbox(char(ME_DiseaseYPred));


  ME_YPredTest_Stage = classify(ME_Diseasenet, ME_DiseaseaugimdsValidation); % Use classify for validation data
  ME_YTest = ME_DiseaseimdsValidation.Labels;

  ME_DiseaseAccuracy = mean(ME_Diseasetraininfo.TrainingAccuracy);

  ME_PSNR_Stage = psnr(img,ME_colorImageOverlay );

  ME_Entropy_Stage = entropy(img);


  [ME_confMat_Stage, ME_order] = confusionmat(ME_YTest, ME_YPredTest_Stage);


  % Calculate metrics
  ME_TP_Stage = diag(ME_confMat_Stage);
  ME_FP_Stage = sum(ME_confMat_Stage, 1)' - ME_TP_Stage;
  ME_FN_Stage = sum(ME_confMat_Stage, 2) - ME_TP_Stage;
  ME_TN_Stage = sum(ME_confMat_Stage(:)) - (ME_TP_Stage + ME_FP_Stage + ME_FN_Stage);

  ME_specificityP_Stage = ME_TN_Stage ./ (ME_TN_Stage + ME_FP_Stage);
  ME_sensitivityP_Stage = ME_TP_Stage ./ (ME_TP_Stage + ME_FN_Stage);  % Sensitivity calculation
  ME_precision_Stage = ME_TP_Stage ./ (ME_TP_Stage + ME_FP_Stage);
  ME_recall_Stage = ME_TP_Stage ./ (ME_TP_Stage + ME_FN_Stage);
  ME_f1_score_Stage = 2 * (ME_precision_Stage .* ME_recall_Stage) ./ (ME_precision_Stage + ME_recall_Stage);
  ME_AUC_Stage = 0.5 * (ME_TP_Stage ./ (ME_TP_Stage + ME_FN_Stage) + ME_TN_Stage ./ (ME_TN_Stage + ME_FP_Stage));
  ME_AUC_Stage = mean(ME_AUC_Stage);

  ME_StageMetrics.Accuracy = ME_DiseaseAccuracy;
  ME_StageMetrics.PSNR = ME_PSNR_Stage;
  ME_StageMetrics.Entropy = ME_Entropy_Stage;
  ME_StageMetrics.AUC = ME_AUC_Stage* 100;
  ME_StageMetrics.Precision = mean(ME_precision_Stage) * 100;
  ME_StageMetrics.Recall = mean(ME_recall_Stage) * 100;
  ME_StageMetrics.F1_Score = mean(ME_f1_score_Stage) * 100;
  ME_StageMetrics.Specificity = mean(ME_specificityP_Stage) * 100;
  ME_StageMetrics.Sensitivity = mean(ME_sensitivityP_Stage) * 100;
  disp(" Stages Macular Edema")
  disp(ME_StageMetrics);

  ME_DiseaseYPred = char(ME_DiseaseYPred);

  if strcmp(ME_DiseaseYPred, 'Mild ME')
    ME_TreatmentOptions = 'Blood sugar control, regular eye exams, anti-inflammatory medications, and observation.';
    ME_LifestyleRecommendations = 'Healthy diet, exercise, stress management, and monitor blood sugar regularly';
  elseif strcmp(ME_DiseaseYPred, 'Moderate ME')
    ME_TreatmentOptions = 'Anti-VEGF injections, corticosteroid treatments, laser therapy, and frequent eye exams';
    ME_LifestyleRecommendations = 'Control blood pressure, avoid smoking, maintain healthy weight, and exercise regularly';
  elseif strcmp(ME_DiseaseYPred, 'Proliferative ME')
    ME_TreatmentOptions = 'Advanced laser treatment, anti-VEGF injections, vitrectomy surgery for severe cases';
    ME_LifestyleRecommendations = 'Monitor eye health, maintain strict diabetes control, and reduce eye strain';
  elseif strcmp(ME_DiseaseYPred, 'Severe ME')
    ME_TreatmentOptions = 'Frequent anti-VEGF injections, vitrectomy, possible retinal surgery, laser treatments';
    ME_LifestyleRecommendations = 'Focus on managing diabetes, regular medical check-ups, reduce physical eye strain';
  else
  end

  generateDiagnosisME_Report(img, Segmented_Image, ME_colorImageOverlay, ME_DiseaseYPred, ME_TreatmentOptions, ME_LifestyleRecommendations, ME_Metrics, ME_StageMetrics)

elseif strcmp(ME_output, 'Healthy')
  ME_DiseaseYPred = 'No Macular Edema';
  ME_TreatmentOptions = ': The patient is healthy, with no signs of Eye Disease';
  ME_LifestyleRecommendations = ' Continue to manage blood sugar levels, follow a healthy diet, engage in regular physical activity, and have regular eye examinations.';
  generateDiagnosisME_Reporthealthy(img, Segmented_Image, ME_DiseaseYPred, ME_TreatmentOptions, ME_LifestyleRecommendations, ME_Metrics)
end
%%
%%
%%
Glucoma_matlabroot = cd;    % Dataset path
Glucoma_datasetpath = fullfile(Glucoma_matlabroot,'Glucoma - Disease Dataset');   %Build full file name from parts
Glucoma_imds = imageDatastore(Glucoma_datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

[Glucoma_imdsTrain, Glucoma_imdsValidation] = splitEachLabel(Glucoma_imds,0.8,'randomized');     %Split ImageDatastore labels by proportions

Glucoma_augimdsTrain = augmentedImageDatastore([224 224 3], Glucoma_imdsTrain);
Glucoma_augimdsValidation = augmentedImageDatastore([224 224 3], Glucoma_imdsValidation);

%%

Glaucoma_net = densenet201

Glucoma_layers = [imageInputLayer([224 224 3])

  Glaucoma_net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers

  fullyConnectedLayer(2) % modifying the fullyconnected layer with respect to classes

  softmaxLayer

  classificationLayer];


Glucoma_options = trainingOptions("sgdm",...
  "ExecutionEnvironment","auto",...
  "InitialLearnRate",0.001,...
  "MaxEpochs",100,...
  "MiniBatchSize",25,...
  "Shuffle","every-epoch",...
  "ValidationData",Glucoma_augimdsValidation, ...
  "ValidationFrequency",50,...
  "Verbose",true, ...
  "Plots","training-progress");


% [Glaucoma_net, Glaucoma_traininfo] = trainNetwork(Glucoma_augimdsTrain,Glucoma_layers,Glucoma_options);  %Train neural network for deep learning

load Glaucoma_net
load Glaucoma_traininfo

Glucoma_YPred = classify(Glaucoma_net,Segmented_Image);
msgbox(char(Glucoma_YPred))

Glucoma_Accuracy = mean(Glaucoma_traininfo.TrainingAccuracy);

% Mertics
Glucoma_YPredTest = classify(Glaucoma_net, Glucoma_augimdsValidation); % Use classify for validation data
Glucoma_Glucoma_YTest = Glucoma_imdsValidation.Labels;

Glucoma_PSNR = psnr(img,Segmented_Image);
Glucoma_Entropy = entropy(img);
[Glucoma_confMat, Glucoma_order] = confusionmat(Glucoma_Glucoma_YTest, Glucoma_YPredTest);


% Calculate metrics
Glucoma_TP = diag(Glucoma_confMat);
Glucoma_FP = sum(Glucoma_confMat, 1)' - Glucoma_TP;
Glucoma_FN = sum(Glucoma_confMat, 2) - Glucoma_TP;
Glucoma_TN = sum(Glucoma_confMat(:)) - (Glucoma_TP + Glucoma_FP + Glucoma_FN);

Glucoma_specificityP = Glucoma_TN ./ (Glucoma_TN + Glucoma_FP);
Glucoma_sensitivityP = Glucoma_TP ./ (Glucoma_TP + Glucoma_FN);
Glucoma_precision = Glucoma_TP ./ (Glucoma_TP + Glucoma_FP);
Glucoma_recall = Glucoma_TP ./ (Glucoma_TP + Glucoma_FN);
Glucoma_f1_score = 2 * (Glucoma_precision .* Glucoma_recall) ./ (Glucoma_precision + Glucoma_recall);
Glucoma_AUC = 0.5 * (Glucoma_TP ./ (Glucoma_TP + Glucoma_FN) + Glucoma_TN ./ (Glucoma_TN + Glucoma_FP));
Glucoma_AUC = mean(Glucoma_AUC);

Glucoma_Metrics.Accuracy = Glucoma_Accuracy;
Glucoma_Metrics.PSNR = Glucoma_PSNR;
Glucoma_Metrics.Entropy = Glucoma_Entropy;
Glucoma_Metrics.AUC = Glucoma_AUC * 100;
Glucoma_Metrics.Precision = mean(Glucoma_precision) * 100;
Glucoma_Metrics.Recall = mean(Glucoma_recall) * 100;
Glucoma_Metrics.F1_Score = mean(Glucoma_f1_score) * 100;
Glucoma_Metrics.Specificity = mean(Glucoma_specificityP) * 100;
Glucoma_Metrics.Sensitivity = mean(Glucoma_sensitivityP) * 100;

disp(" Glaucoma")
disp(Glucoma_Metrics);

%%
Glucoma_output = char(Glucoma_YPred);

if strcmp(Glucoma_output, 'Glaucoma')
  %% Image Processing Techniques
  % Step 1: Morphological Closing Operation
  Glucoma_se = strel('disk', 10); % Create a disk-shaped structuring element with radius 10
  Glucoma_closeBW = imclose(img, Glucoma_se); % Perform morphological closing operation

  % Step 2: Adaptive Contrast Enhancement using LAB color space
  Glucoma_shadow_lab = rgb2lab(Glucoma_closeBW); % Convert the closed image to LAB color space
  Glucoma_max_luminosity = 100;
  Glucoma_L = Glucoma_shadow_lab(:,:,1)/Glucoma_max_luminosity; % Extract the L (luminosity) channel

  Glucoma_shadow_imadjust = Glucoma_shadow_lab;
  Glucoma_shadow_imadjust(:,:,1) = imadjust(Glucoma_L) * Glucoma_max_luminosity; % Apply contrast enhancement on the L channel
  Glucoma_shadow_imadjust = lab2rgb(Glucoma_shadow_imadjust); % Convert back to RGB

  % Step 3: Gabor Filter Application
  Glucoma_I = rgb2gray(Glucoma_shadow_imadjust); % Convert enhanced image to grayscale
  Glucoma_wavelength = 4; % Set the wavelength for the Gabor filter
  Glucoma_orientation = 90; % Set the orientation for the Gabor filter
  [Glucoma_mag, Glucoma_phase] = imgaborfilt(Glucoma_I, Glucoma_wavelength, Glucoma_orientation); % Apply Gabor filter


  Glucoma_level = graythresh(Glucoma_phase); % Compute Otsu's threshold level
  Glucoma_BW = imbinarize(Glucoma_phase, Glucoma_level); % Binarize the phase image using the threshold

  % Step 5: Overlay Binary Map on Original Image
  Glucoma_colorImageOverlay = imoverlay(img, Glucoma_BW, [1 0 0]); % Overlay binary map on the original image in red


  Glucoma_Diseasematlabroot = cd;    % Dataset path
  Glucoma_Diseasedatasetpath = fullfile(Glucoma_Diseasematlabroot,'Glucoma - DiseaseStage Dataset');   %Build full file name from parts
  Glucoma_Diseaseimds = imageDatastore(Glucoma_Diseasedatasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

  [Glucoma_DiseaseimdsTrain, Glucoma_DiseaseimdsValidation] = splitEachLabel(Glucoma_Diseaseimds,0.8);     %Split ImageDatastore labels by proportions

  Glucoma_DiseaseaugimdsTrain = augmentedImageDatastore([224 224 3],Glucoma_DiseaseimdsTrain);  %Generate batches of augmented image data
  Glucoma_DiseaseaugimdsValidation = augmentedImageDatastore([224 224 3],Glucoma_DiseaseimdsValidation);
  % Training Options
  Glucoma_Diseaseoptions = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",100,...
    "MiniBatchSize",25,...
    "Shuffle","every-epoch",...
    "ValidationData",Glucoma_DiseaseaugimdsTrain, ...
    "ValidationFrequency",50,...
    "Verbose",true, ...
    "Plots","training-progress");

  Glucoma_Diseaselayers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

  %       [Glucoma_Diseasenet, Glucoma_Diseasetraininfo] = trainNetwork(Glucoma_DiseaseaugimdsTrain,Glucoma_Diseaselayers,Glucoma_Diseaseoptions);  %Train neural network for deep learning

  load Glucoma_Diseasenet
  load Glucoma_Diseasetraininfo

  [Glucoma_DiseaseYPred,Glucoma_Diseasescore] = classify(Glucoma_Diseasenet,Glucoma_colorImageOverlay);      %Classify data using a trained deep learning neural network
  msgbox(char(Glucoma_DiseaseYPred));


  Glucoma_YPredTest_Stage = classify(Glucoma_Diseasenet, Glucoma_DiseaseaugimdsValidation); % Use classify for validation data
  Glucoma_Glucoma_YTest = Glucoma_DiseaseimdsValidation.Labels;

  Glucoma_DiseaseAccuracy = mean(Glucoma_Diseasetraininfo.TrainingAccuracy);

  Glucoma_PSNR_Stage = psnr(img,Glucoma_colorImageOverlay );

  Glucoma_Entropy_Stage = entropy(img);
  % Generate confusion matrix
  Glucoma_confMat_Stage = confusionmat(Glucoma_Glucoma_YTest, Glucoma_YPredTest_Stage);
  Glucoma_TN_Stage = Glucoma_confMat_Stage(1, 1);
  Glucoma_FP_Stage = Glucoma_confMat_Stage(1, 2);
  Glucoma_FN_Stage = Glucoma_confMat_Stage(2, 1);
  Glucoma_TP_Stage = Glucoma_confMat_Stage(2, 2);

  Glucoma_specificityP_Stage = Glucoma_TP_Stage / (Glucoma_TP_Stage + Glucoma_FN_Stage);
  Glucoma_sensitivityP_Stage = Glucoma_TN_Stage / (Glucoma_TN_Stage + Glucoma_FP_Stage);

  Glucoma_precision_Stage = Glucoma_TP_Stage / (Glucoma_TP_Stage + Glucoma_FP_Stage);
  Glucoma_recall_Stage = Glucoma_specificityP_Stage;

  Glucoma_f1_score_Stage = 2 * (Glucoma_precision_Stage * Glucoma_recall_Stage) / ...
    (Glucoma_precision_Stage + Glucoma_recall_Stage);

  Glucoma_AUC_Stage = 0.5 * (Glucoma_TP_Stage / (Glucoma_TP_Stage + Glucoma_FN_Stage) + ...
    Glucoma_TN_Stage / (Glucoma_TN_Stage + Glucoma_FP_Stage));


  Glucoma_StageMetrics.Accuracy = Glucoma_DiseaseAccuracy;
  Glucoma_StageMetrics.PSNR = Glucoma_PSNR_Stage;
  Glucoma_StageMetrics.Entropy = Glucoma_Entropy_Stage;
  Glucoma_StageMetrics.AUC = Glucoma_AUC_Stage * 100;
  Glucoma_StageMetrics.Precision = mean(Glucoma_precision_Stage) * 100;
  Glucoma_StageMetrics.Recall = mean(Glucoma_recall_Stage) * 100;
  Glucoma_StageMetrics.F1_Score = mean(Glucoma_f1_score_Stage) * 100;
  Glucoma_StageMetrics.Specificity = mean(Glucoma_specificityP_Stage) * 100;
  Glucoma_StageMetrics.Sensitivity = mean(Glucoma_sensitivityP_Stage) * 100;
  disp(" Stages Glaucoma")
  disp(Glucoma_StageMetrics);

  Glucoma_DiseaseYPred = char(Glucoma_DiseaseYPred);

  if strcmp(Glucoma_DiseaseYPred, 'Mild Glaucoma')
    Glucoma_TreatmentOptions = 'Prescription eye drops to lower intraocular pressure, regular eye exams';
    Glucoma_LifestyleRecommendations = 'Maintain healthy diet, avoid caffeine, regular exercise, reduce stress, monitor';
  elseif strcmp(Glucoma_DiseaseYPred, 'Moderate Glaucoma')
    Glucoma_TreatmentOptions = 'Eye drops, laser therapy for pressure reduction, ongoing ophthalmologist consultations';
    Glucoma_LifestyleRecommendations = 'Follow treatment schedule, avoid smoking, wear protective eyewear, healthy lifestyle';
  elseif strcmp(Glucoma_DiseaseYPred, 'Proliferative Glaucoma')
    Glucoma_TreatmentOptions = 'Advanced laser therapy, surgical procedures, medication for pressure management, monitoring';
    Glucoma_LifestyleRecommendations = 'Strict treatment adherence, avoid eye strain, manage systemic health conditions';
  elseif strcmp(Glucoma_DiseaseYPred, 'Severe Glaucoma')
    Glucoma_TreatmentOptions = 'Surgical interventions like trabeculectomy, drainage devices, or laser surgery required';
    Glucoma_LifestyleRecommendations = 'Monitor vision changes, maintain regular checkups, manage stress, avoid overexertion';
  else
  end
  generateDiagnosisGlaucoma_Report(img, Segmented_Image, Glucoma_colorImageOverlay, Glucoma_DiseaseYPred, Glucoma_TreatmentOptions, Glucoma_LifestyleRecommendations, Glucoma_Metrics, Glucoma_StageMetrics)

elseif strcmp(Glucoma_output, 'Healthy')
  Glucoma_DiseaseYPred = 'No Glaucoma';
  Glucoma_TreatmentOptions = ': The patient is healthy, with no signs of Glaucoma';
  Glucoma_LifestyleRecommendations = ' Continue to manage blood sugar levels, follow a healthy diet, engage in regular physical activity, and have regular eye examinations.';
  generateDiagnosisGlaucoma_Reporthealthy(img, Segmented_Image, Glucoma_DiseaseYPred, Glucoma_TreatmentOptions, Glucoma_LifestyleRecommendations, Glucoma_Metrics)
end
%%
%%
%%
Exudates_matlabroot = cd;    % Dataset path
Exudates_datasetpath = fullfile(Exudates_matlabroot,'Exudates - Disease Dataset');   %Build full file name from parts
Exudates_imds = imageDatastore(Exudates_datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

[Exudates_imdsTrain, Exudates_imdsValidation] = splitEachLabel(Exudates_imds,0.8,'randomized');     %Split ImageDatastore labels by proportions

Exudates_augimdsTrain = augmentedImageDatastore([224 224 3], Exudates_imdsTrain);
Exudates_augimdsValidation = augmentedImageDatastore([224 224 3], Exudates_imdsValidation);


Exudates_net = densenet201

Exudates_layers = [imageInputLayer([224 224 3])

  Exudates_net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers

  fullyConnectedLayer(2) % modifying the fullyconnected layer with respect to classes

  softmaxLayer

  classificationLayer];


Exudates_options = trainingOptions("sgdm",...
  "ExecutionEnvironment","auto",...
  "InitialLearnRate",0.001,...
  "MaxEpochs",100,...
  "MiniBatchSize",25,...
  "Shuffle","every-epoch",...
  "ValidationData",Exudates_augimdsValidation, ...
  "ValidationFrequency",50,...
  "Verbose",true, ...
  "Plots","training-progress");


% [Exudates_net, Exudates_traininfo] = trainNetwork(Exudates_augimdsTrain,Exudates_layers,Exudates_options);  %Train neural network for deep learning

load Exudates_net
load Exudates_traininfo

Exudates_YPred = classify(Exudates_net,Segmented_Image);
msgbox(char(Exudates_YPred))

Exudates_Accuracy = mean(Exudates_traininfo.TrainingAccuracy);

% Mertics
Exudates_YPredTest = classify(Exudates_net, Exudates_augimdsValidation); % Use classify for validation data
Exudates_Exudates_YTest = Exudates_imdsValidation.Labels;

Exudates_PSNR = psnr(img,Segmented_Image);
Exudates_Entropy = entropy(img);
[Exudates_confMat, Exudates_Exudates_order] = confusionmat(Exudates_Exudates_YTest, Exudates_YPredTest);

% Calculate metrics
Exudates_TP = diag(Exudates_confMat);
Exudates_FP = sum(Exudates_confMat, 1)' - Exudates_TP;
Exudates_FN = sum(Exudates_confMat, 2) - Exudates_TP;
Exudates_TN = sum(Exudates_confMat(:)) - (Exudates_TP + Exudates_FP + Exudates_FN);

Exudates_specificityP = Exudates_TN ./ (Exudates_TN + Exudates_FP);
Exudates_sensitivityP = Exudates_TP ./ (Exudates_TP + Exudates_FN);
Exudates_precision = Exudates_TP ./ (Exudates_TP + Exudates_FP);
Exudates_recall = Exudates_TP ./ (Exudates_TP + Exudates_FN);
Exudates_f1_score = 2 * (Exudates_precision .* Exudates_recall) ./ (Exudates_precision + Exudates_recall);
Exudates_AUC = 0.5 * (Exudates_TP ./ (Exudates_TP + Exudates_FN) + Exudates_TN ./ (Exudates_TN + Exudates_FP));
Exudates_AUC = mean(Exudates_AUC);

Exudates_Metrics.Accuracy = Exudates_Accuracy;
Exudates_Metrics.PSNR = Exudates_PSNR;
Exudates_Metrics.Entropy = Exudates_Entropy;
Exudates_Metrics.AUC = Exudates_AUC * 100;
Exudates_Metrics.Precision = mean(Exudates_precision) * 100;
Exudates_Metrics.Recall = mean(Exudates_recall) * 100;
Exudates_Metrics.F1_Score = mean(Exudates_f1_score) * 100;
Exudates_Metrics.Specificity = mean(Exudates_specificityP) * 100;
Exudates_Metrics.Sensitivity = mean(Exudates_sensitivityP) * 100;
disp(" Exudates")
disp(Exudates_Metrics);

%%
Exudates_output = char(Exudates_YPred);

if strcmp(Exudates_output, 'Exudates')
  %% Image Processing Techniques
  % Step 1: Morphological Closing Operation
  Exudates_se = strel('disk', 10); % Create a disk-shaped structuring element with radius 10
  Exudates_closeBW = imclose(img, Exudates_se); % Perform morphological closing operation


  % Step 2: Adaptive Contrast Enhancement using LAB color space
  Exudates_shadow_lab = rgb2lab(Exudates_closeBW); % Convert the closed image to LAB color space
  Exudates_max_luminosity = 100;
  Exudates_L = Exudates_shadow_lab(:,:,1)/Exudates_max_luminosity; % Extract the L (luminosity) channel

  Exudates_shadow_imadjust = Exudates_shadow_lab;
  Exudates_shadow_imadjust(:,:,1) = imadjust(Exudates_L) * Exudates_max_luminosity; % Apply contrast enhancement on the L channel
  Exudates_shadow_imadjust = lab2rgb(Exudates_shadow_imadjust); % Convert back to RGB

  % Step 3: Gabor Filter Application
  Exudates_I = rgb2gray(Exudates_shadow_imadjust); % Convert enhanced image to grayscale
  Exudates_wavelength = 4; % Set the wavelength for the Gabor filter
  Exudates_orientation = 90; % Set the orientation for the Gabor filter
  [Exudates_mag, Exudates_phase] = imgaborfilt(Exudates_I, Exudates_wavelength, Exudates_orientation); % Apply Gabor filter

  Exudates_level = graythresh(Exudates_phase); % Compute Otsu's threshold level
  Exudates_BW = imbinarize(Exudates_phase, Exudates_level); % Binarize the phase image using the threshold

  % Step 5: Overlay Binary Map on Original Image
  Exudates_colorImageOverlay = imoverlay(img, Exudates_BW, [1 0 0]); % Overlay binary map on the original image in red

  Exudates_Diseasematlabroot = cd;    % Dataset path
  Exudates_Diseasedatasetpath = fullfile(Exudates_Diseasematlabroot,'Exudates - DiseaseStage Dataset');   %Build full file name from parts
  Exudates_Diseaseimds = imageDatastore(Exudates_Diseasedatasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

  [Exudates_DiseaseimdsTrain, Exudates_DiseaseimdsValidation] = splitEachLabel(Exudates_Diseaseimds,0.8);     %Split ImageDatastore labels by proportions

  Exudates_DiseaseaugimdsTrain = augmentedImageDatastore([224 224 3],Exudates_DiseaseimdsTrain);  %Generate batches of augmented image data
  Exudates_DiseaseaugimdsValidation = augmentedImageDatastore([224 224 3],Exudates_DiseaseimdsValidation);
  % Training Options
  Exudates_Diseaseoptions = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",100,...
    "MiniBatchSize",25,...
    "Shuffle","every-epoch",...
    "ValidationData",Exudates_DiseaseaugimdsTrain, ...
    "ValidationFrequency",50,...
    "Verbose",true, ...
    "Plots","training-progress");

  Exudates_Diseaselayers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

  %       [Exudates_Diseasenet, Exudates_Diseasetraininfo] = trainNetwork(Exudates_DiseaseaugimdsTrain,Exudates_Diseaselayers,Exudates_Diseaseoptions);  %Train neural network for deep learning

  load Exudates_Diseasenet
  load Exudates_Diseasetraininfo
  %
  [Exudates_Exudates_DiseaseYPred,Exudates_Diseasescore] = classify(Exudates_Diseasenet,Exudates_colorImageOverlay);      %Classify data using a trained deep learning neural network
  msgbox(char(Exudates_Exudates_DiseaseYPred));


  Exudates_YPredTest_Stage = classify(Exudates_Diseasenet, Exudates_DiseaseaugimdsValidation); % Use classify for validation data
  Exudates_Exudates_YTest = Exudates_DiseaseimdsValidation.Labels;

  Exudates_DiseaseAccuracy = mean(Exudates_Diseasetraininfo.TrainingAccuracy);

  Exudates_PSNR_Stage = psnr(img,Exudates_colorImageOverlay );

  Exudates_Entropy_Stage = entropy(img);
  Exudates_confMat_Stage = confusionmat(Exudates_Exudates_YTest, Exudates_YPredTest_Stage);
  Exudates_TN_Stage = Exudates_confMat_Stage(1, 1);
  Exudates_FP_Stage = Exudates_confMat_Stage(1, 2);
  Exudates_FN_Stage = Exudates_confMat_Stage(2, 1);
  Exudates_TP_Stage = Exudates_confMat_Stage(2, 2);

  sensitivity = Exudates_TP_Stage / (Exudates_TP_Stage + Exudates_FN_Stage);
  specificity = Exudates_TN_Stage / (Exudates_TN_Stage + Exudates_FP_Stage);

  Exudates_precision_Stage = Exudates_TP_Stage / (Exudates_TP_Stage + Exudates_FP_Stage);

  Exudates_recall_Stage = sensitivity;

  Exudates_f1_score_Stage = 2 * (Exudates_precision_Stage * Exudates_recall_Stage) / ...
    (Exudates_precision_Stage + Exudates_recall_Stage);
  Exudates_AUC_Stage = 0.5 * (Exudates_TP_Stage / (Exudates_TP_Stage + Exudates_FN_Stage) + ...
    Exudates_TN_Stage / (Exudates_TN_Stage + Exudates_FP_Stage));

  Exudates_StageMetrics.Accuracy = Exudates_DiseaseAccuracy; % Ensure Exudates_DiseaseAccuracy is defined
  Exudates_StageMetrics.PSNR = Exudates_PSNR_Stage; % Ensure Exudates_PSNR_Stage is defined
  Exudates_StageMetrics.Entropy = Exudates_Entropy_Stage; % Ensure Exudates_Entropy_Stage is defined
  Exudates_StageMetrics.AUC = mean(Exudates_AUC_Stage) * 100;
  Exudates_StageMetrics.Precision = mean(Exudates_precision_Stage) * 100;
  Exudates_StageMetrics.Recall = mean(Exudates_recall_Stage) * 100;
  Exudates_StageMetrics.F1_Score = mean(Exudates_f1_score_Stage) * 100;
  Exudates_StageMetrics.Specificity = mean(specificity) * 100;
  Exudates_StageMetrics.Sensitivity = mean(sensitivity) * 100;
  disp(" Stages Exudates")
  disp(Exudates_StageMetrics);
  Exudates_Exudates_DiseaseYPred = char(Exudates_Exudates_DiseaseYPred);

  if strcmp(Exudates_Exudates_DiseaseYPred, 'Mild Exudates')
    Exudates_TreatmentOptions = 'Monitor closely, control blood sugar, anti-inflammatory medications, regular eye exams';
    Exudates_LifestyleRecommendations = 'Healthy diet, exercise, manage diabetes, avoid smoking, monitor eye health';
  elseif strcmp(Exudates_Exudates_DiseaseYPred, 'Moderate Exudates')
    Exudates_TreatmentOptions = 'Laser photocoagulation, anti-VEGF injections, control systemic conditions, regular monitoring';
    Exudates_LifestyleRecommendations = 'Strict diabetes management, low-sodium diet, regular exercise, stress management routin';
  elseif strcmp(Exudates_Exudates_DiseaseYPred, 'Proliferative Exudates')
    Exudates_TreatmentOptions = 'Frequent anti-VEGF injections, advanced laser therapy, consider corticosteroid treatments';
    Exudates_LifestyleRecommendations = 'Monitor closely, maintain treatment adherence, avoid eye strain, regular consultations';
  elseif strcmp(Exudates_Exudates_DiseaseYPred, 'Severe Exudates')
    Exudates_TreatmentOptions = 'Vitrectomy surgery, aggressive anti-VEGF therapy, advanced laser photocoagulation required';
    Exudates_LifestyleRecommendations = 'Focus on diabetes management, regular checkups, low-impact exercise, emotional support';
  else
  end
  generateDiagnosisExudates_Report(img, Segmented_Image, Exudates_colorImageOverlay, Exudates_Exudates_DiseaseYPred, Exudates_TreatmentOptions, Exudates_LifestyleRecommendations, Exudates_Metrics, Exudates_StageMetrics)

elseif strcmp(Exudates_output, 'Healthy')
  Exudates_Exudates_DiseaseYPred = 'No Exudates';
  Exudates_TreatmentOptions = ': The patient is healthy, with no signs of Exudates';
  Exudates_LifestyleRecommendations = ' Continue to manage blood sugar levels, follow a healthy diet, engage in regular physical activity, and have regular eye examinations.';
  generateDiagnosisExudates_Reporthealthy(img, Segmented_Image, Exudates_Exudates_DiseaseYPred, Exudates_TreatmentOptions, Exudates_LifestyleRecommendations,Exudates_Metrics)
end

%% Send the email with the PDF attachments
senderEmail = '';  % Your Gmail email address
appPassword = '';    % Your Gmail App Password
recipientEmail = '';
subject = 'Medical Diagnosis Report';

% Set up the SMTP server settings for Gmail
setpref('Internet', 'E_mail', senderEmail);
setpref('Internet', 'SMTP_Server', 'smtp.gmail.com');
setpref('Internet', 'SMTP_Username', senderEmail);
setpref('Internet', 'SMTP_Password', appPassword);

% Gmail uses port 587 for TLS
props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth', 'true');
props.setProperty('mail.smtp.port', '587');
props.setProperty('mail.smtp.starttls.enable', 'true');

% Specify the PDF file names
pdfFiles = {'DiagnosisReport_Diabetic_Retinopathy.pdf', 'DiagnosisReport_Exudates.pdf', 'DiagnosisReport_GLAUCOMA.pdf', 'DiagnosisReport_Macular_Edema.pdf'};

try
  % Send email with multiple PDF attachments
  sendmail(recipientEmail, subject, '', pdfFiles); % No additional content in the email
  disp('Email with medical reports sent successfully.');
  disp('Please check your email.');
catch
  disp('Error sending email. Check your credentials and internet connection.');
end

%%

YPred = char(YPred); % Ensure YPred is a character array

if strcmp(YPred, 'Healthy')
  data1 = 0;  % Numeric value for Healthy
  data2 = NaN;
elseif strcmp(YPred, 'Diabetic Retinopathy')
  data1 = 1;  % Numeric value for Diabetic Retinopathy

  DiseaseYPred = char(DiseaseYPred);

  if strcmp(DiseaseYPred, 'Mild DR')
    data2 = 0;
  elseif strcmp(DiseaseYPred, 'Moderate DR')
    data2 = 1;
  elseif strcmp(DiseaseYPred, 'Proliferative DR')
    data2 = 2;
  elseif strcmp(DiseaseYPred, 'Severe DR')
    data2 = 3;
  else
    data2 = NaN;
  end
end


ME_YPred = char(ME_YPred); % Ensure YPred is a character array

if strcmp(ME_YPred, 'Healthy')
  data3 = 0;  % Numeric value for Healthy
  data4 = NaN;
elseif strcmp(ME_YPred, 'Macular Edema')
  data3 = 1;  % Numeric value for Diabetic Retinopathy

  ME_DiseaseYPred = char(ME_DiseaseYPred);

  if strcmp(ME_DiseaseYPred, 'Mild ME')
    data4 = 0;
  elseif strcmp(ME_DiseaseYPred, 'Moderate ME')
    data4 = 1;
  elseif strcmp(ME_DiseaseYPred, 'Proliferative ME')
    data4 = 2;
  elseif strcmp(ME_DiseaseYPred, 'Severe ME')
    data4 = 3;
  else
    data4 = NaN;
  end
end

Glucoma_YPred = char(Glucoma_YPred); % Ensure YPred is a character array

if strcmp(Glucoma_YPred, 'Healthy')
  data5 = 0;  % Numeric value for Healthy
  data6 = NaN;
elseif strcmp(Glucoma_YPred, 'Glaucoma')
  data5 = 1;  % Numeric value for Diabetic Retinopathy

  Glucoma_DiseaseYPred = char(Glucoma_DiseaseYPred);

  if strcmp(Glucoma_DiseaseYPred, 'Mild Glaucoma')
    data6 = 0;
  elseif strcmp(Glucoma_DiseaseYPred, 'Moderate Glaucoma')
    data6 = 1;
  elseif strcmp(Glucoma_DiseaseYPred, 'Proliferative Glaucoma')
    data6 = 2;
  elseif strcmp(Glucoma_DiseaseYPred, 'Severe Glaucoma')
    data6 = 3;
  else
    data6 = NaN;
  end
end


Exudates_YPred = char(Exudates_YPred); % Ensure YPred is a character array

if strcmp(Exudates_YPred, 'Healthy')
  data7 = 0;  % Numeric value for Healthy
  data8 = NaN;
elseif strcmp(Exudates_YPred, 'Exudates')
  data7 = 1;  % Numeric value for Diabetic Retinopathy

  Exudates_Exudates_DiseaseYPred = char(Exudates_Exudates_DiseaseYPred);

  if strcmp(Exudates_Exudates_DiseaseYPred, 'Mild Exudates')
    data8 = 0;
  elseif strcmp(Exudates_Exudates_DiseaseYPred, 'Moderate Exudates')
    data8 = 1;
  elseif strcmp(Exudates_Exudates_DiseaseYPred, 'Proliferative Exudates')
    data8 = 2;
  elseif strcmp(Exudates_Exudates_DiseaseYPred, 'Severe Exudates')
    data8 = 3;
  else
    data8 = NaN;
  end
end


Channel_ID = 2755090;            % Replace with your channel ID
Write_API_Key = 'JF377LHSR6M0IYNQ'; % Replace with your Write API Key

% Send the data to ThingSpeak
thingSpeakWrite(Channel_ID, [data1, data2, data3, data4, data5, data6, data7, data8], 'Fields', [1, 2, 3, 4, 5, 6, 7, 8], 'WriteKey', Write_API_Key);

disp('Data sent to ThingSpeak successfully!');
%%
% (Copy the entire content of the provided MATLAB script here, starting from the preprocessing step)

% At the end of the script, format the results as JSON for easy parsing in Node.js
results = struct();
results.dr = struct('detected', strcmp(YPred, 'Diabetic Retinopathy'), 'severity', DiseaseYPred, 'metrics', Metrics);
results.me = struct('detected', strcmp(ME_YPred, 'Macular Edema'), 'severity', ME_DiseaseYPred, 'metrics', ME_Metrics);
results.glaucoma = struct('detected', strcmp(Glucoma_YPred, 'Glaucoma'), 'severity', Glucoma_DiseaseYPred, 'metrics', Glucoma_Metrics);
results.exudates = struct('detected', strcmp(Exudates_YPred, 'Exudates'), 'severity', Exudates_Exudates_DiseaseYPred, 'metrics', Exudates_Metrics);

% Convert to JSON and print to stdout
jsonStr = jsonencode(results);
fprintf(1, '%s', jsonStr);

