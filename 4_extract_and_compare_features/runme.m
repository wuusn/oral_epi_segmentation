
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This a script for extracting features from histological image


%   
% (c) Edited by  Yu Zhou, George Lee, and Cheng Lu
% Biomedical Engineering,
% Case Western Reserve Univeristy, cleveland, OH. Aug, 2017
% If you have any problem feel free to contact me.
% Please address questions or comments to: hacylu@yahoo.com

% Terms of use: You are free to copy,
% distribute, display, and use this work, under the following
% conditions. (1) You must give the original authors credit. (2) You may
% not use or redistribute this work for commercial purposes. (3) You may
% not alter, transform, or build upon this work. (4) For any reuse or
% distribution, you must make clear to others the license terms of this
% work. (5) Any of these conditions can be waived if you get permission
% from the authors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dbstop if error
addpath(genpath(pwd));
% addpath(genpath('./CCM_CRL'));
% C:\Nutstore\Nutstore\4PublicProgram\Feature_extraction\extract_all_features\FeatureExtraction\CCM_CRL
%% step 1: load image
img_path = 'SP13-4226 D2.png';
mask_path = 'SP13-4226 D2_pred.png';
img = imread(img_path);
mask=imread(mask_path);
datetime()
%% step 2: segment nuclei and save boundaries, change the scale for different magnification of image
% scales = 8:2:14;
scales = 8:4:16;
[~, nuclei, properties] = Veta(img, scales);
bounds=Lmask2bounds(mask);
datetime()
%% step 3: extract features
% 1. graph features(51)
%[graphfeats] = extract_all_features(bounds,img,1);

% 2. morphological features(100)
[morphfeats] = extract_all_features(bounds, img, 2);

% 3. CGT features(39)
%[CGTfeats] = extract_all_features(bounds, img, 3);

% 4. cluster graph features(26)
%[clustergraphfeats] = extract_all_features(bounds, img, 4);

% you donn't want to calculate texture features I guess :) it is time
% comsuming and may end up with nothing useful
% 5. haralick features(26) 
[haralickfeats] = extract_all_features(bounds, img, 5);
datetime()
% % 6. texture features (720) (Grayscale(15), Gabor(24), Laws(25), Local Binary Pattern(16) = 80 * (HSV 3channels) * (mean, std, mode))
% [texturefeats] = extract_texture_features(img);

