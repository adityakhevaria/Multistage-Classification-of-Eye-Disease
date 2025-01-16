clear all;close all;clc;
srcFiles = dir('Y:\OTHER WORKS\2024 - 2025\BUSINESS PROJECTS\OCTOMBER - 2024\TK159539  - Internet of Things and Deep Learning Enabled Diabetic Retinopathy Diagnosis Using Retinal Fundus Images\MODIFICATIONS\CODE\DATASET - DR\Severe DR\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
filename = strcat('Y:\OTHER WORKS\2024 - 2025\BUSINESS PROJECTS\OCTOMBER - 2024\TK159539  - Internet of Things and Deep Learning Enabled Diabetic Retinopathy Diagnosis Using Retinal Fundus Images\MODIFICATIONS\CODE\DATASET - DR\Severe DR\',srcFiles(i).name);
img = imread(filename);

img = imresize(img, [224 224]);

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


newfilename=fullfile('Y:\OTHER WORKS\2024 - 2025\BUSINESS PROJECTS\OCTOMBER - 2024\TK159539  - Internet of Things and Deep Learning Enabled Diabetic Retinopathy Diagnosis Using Retinal Fundus Images\MODIFICATIONS\CODE\DR - DiseaseStage Dataset\Severe DR\',srcFiles(i).name);
imwrite(colorImageOverlay,newfilename,'png');
end

close all