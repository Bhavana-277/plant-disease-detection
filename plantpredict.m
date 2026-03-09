function [disease, severityPercent, treatment] = plantpredict(imagePath)

%% ---------------- LOAD MODEL ----------------
load('C:\Users\Bhavana\OneDrive\Documents\PlantDiseaseWebsite\net.mat');
% Make sure path to net.mat is correct

%% ---------------- READ IMAGE ----------------
img = imread(imagePath);
img = imresize(img,[256 256]);

%% -------- Disease Prediction --------
YPred = classify(net,img);

disease = lower(char(string(YPred)));
disease = replace(disease,'___',' ');
disease = replace(disease,'_',' ');

%% -------- Severity Detection --------
hsvImg = rgb2hsv(img);
H = hsvImg(:,:,1);
S = hsvImg(:,:,2);

% Leaf segmentation
leafMask = (H>0.15 & H<0.45) & (S>0.2);
leafMask = imfill(leafMask,'holes');
leafMask = bwareaopen(leafMask,500);

% Disease detection
grayImg = rgb2gray(img);
grayImg = imadjust(grayImg);
diseaseMask = imbinarize(grayImg,'adaptive','Sensitivity',0.55);
diseaseMask = bwareaopen(diseaseMask,80);
diseaseMask = diseaseMask & leafMask;

%% -------- Severity Calculation --------
leafPixels = sum(leafMask(:));
diseasePixels = sum(diseaseMask(:));

if leafPixels == 0
    severityPercent = 0;
else
    severityPercent = (diseasePixels/leafPixels)*100;
end

%% -------- Treatment Database --------
diseaseList = {
'black rot','Prune infected branches and apply fungicide.';
'apple scab','Apply fungicide and remove infected leaves.';
'early blight','Apply fungicide spray.';
'late blight','Remove infected plants.';
'powdery mildew','Apply sulfur fungicide.';
'healthy','Plant is healthy. No treatment required.';
};

treatment = 'Treatment not available';

for i = 1:size(diseaseList,1)
    if contains(disease,diseaseList{i,1})
        treatment = diseaseList{i,2};
        break;
    end
end

%% -------- DISPLAY RESULT --------
fprintf('\n===== PLANT DISEASE RESULT =====\n');
fprintf('Disease   : %s\n', disease);
fprintf('Severity  : %.2f %%\n', severityPercent);
fprintf('Treatment : %s\n', treatment);

end
