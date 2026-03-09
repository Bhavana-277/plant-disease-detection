function [disease,severityLevel,severityPercent,solution,leafMask,diseaseMask] = predict(imagePath)
warning off;

%% Read Image
img = imread(imagePath);
img_resize = imresize(img,[256 256]);

%% Load trained network
load net

%% Disease Prediction
YPred = classify(net,img_resize);

disease = char(string(YPred));
disease = lower(disease);
disease = replace(disease,'___',' ');
disease = replace(disease,'_',' ');

%% Severity Detection
hsvImg = rgb2hsv(img_resize);

H = hsvImg(:,:,1);
S = hsvImg(:,:,2);

leafMask = (H > 0.15 & H < 0.45) & (S > 0.2);
leafMask = imfill(leafMask,'holes');
leafMask = bwareaopen(leafMask,500);

grayImg = rgb2gray(img_resize);
grayImg = imadjust(grayImg);

diseaseMask = imbinarize(grayImg,'adaptive','Sensitivity',0.55);
diseaseMask = bwareaopen(diseaseMask,80);

diseaseMask = diseaseMask & leafMask;

leafPixels = sum(leafMask(:));
diseasePixels = sum(diseaseMask(:));

severityPercent = (diseasePixels / leafPixels) * 100;

if severityPercent < 10
    severityLevel = 'Mild';
elseif severityPercent < 30
    severityLevel = 'Moderate';
else
    severityLevel = 'Severe';
end

solution = "Apply fungicide and remove infected leaves.";
% Convert logical masks to uint8 images
leafMask = uint8(leafMask) * 255;
diseaseMask = uint8(diseaseMask) * 255;


end
