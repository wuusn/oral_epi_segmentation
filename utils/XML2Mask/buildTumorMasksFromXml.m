clear;clc;

%imgFolder='Z:\data\CCF_OropharyngealCarcinoma\Ventana\';
%imgFolder='Z:\data\Kaisar_OP\Ventana_KA_Slides\';
imgFolder='/mnt/md0/_datasets/OralCavity/WSI/SFVA/';
%imgFolder='/mnt/md0/_datasets/OralCavity/WSI/UCSF/';
%imgFolder='/mnt/md0/_datasets/OralCavity/WSI/Vanderbilt/';
%annotFolder='D:\German\Data\Oroph_CCF\annot_patoroc\';
%annotFolder='D:\German\Data\Oroph_CCF\annotations\lymphoid_tissue_Paula\';
%annotFolder='D:\German\Data\Oroph_Kaisar\xml_annot\';
annotFolder='/mnt/md0/_datasets/OralCavity/WSI/SFVA/Annotations_SFVA/';
%annotFolder='/mnt/md0/_datasets/OralCavity/WSI/Vanderbilt/annotations_epi_masks_Vanderbilt/';
%annotFolder='/mnt/md0/_datasets/OralCavity/WSI/UCSF/annotations_epi_tum_nontum_UCSF/';

%tissueMaskFolder='D:\German\Data\Oroph_CCF\masks\tissue_masks\';
%tissueMaskFolder='D:\German\Data\Oroph_Kaisar\masks\tissue_masks\';
%tissueMaskFolder = imgFolder;
%tissueMaskExt='_pred_milesialx2_cb_17.png'; >2**32

%outFolder='D:\German\Data\Oroph_CCF\masks\tumor_masks\';
%outFolder='D:\German\Data\Oroph_CCF\masks\lymphoid_tissue_masks\';
%outFolder='D:\German\Data\Oroph_Kaisar\masks\tumor_masks\';
outFolder='/mnt/md0/_datasets/OralCavity/WSI/SFVA/Masks/tumor/';
%outFolder='/mnt/md0/_datasets/OralCavity/WSI/Vanderbilt/Masks/blue/';
%outFolder='/mnt/md0/_datasets/OralCavity/WSI/UCSF/Masks/blue/';

files=dir([annotFolder '*.xml']);
numFiles=length(files);

%%-- Negative list: Only non-tumor areas were annotated
negativeList={};
%negativeList={'CCFOP20','CCFOP21','CCFOP22','CCFOP24','CCFOP25','CCFOP27',...
%    'CCFOP30','CCFOP31','CCFOP34','CCFOP37','CCFOP42','CCFOP43','CCFOP45',...
%    'CCFOP47','CCFOP54','CCFOP56','CCFOP58','CCFOP60','CCFOP63','CCFOP72',...
%    'CCFOP75','CCFOP77','CCFOP78','CCFOP80',};
%onlyList={'UCSF-OC; C24','UCSF-OC; C28','UCSF-OC; C29','UCSF-OS; K34'};
%onlyList={'OTC-139-D','OTC-136-D','OTC-131-D'};
onlyList={'SP08-1469 E2', 'SP08-1469 E3'};
for i=1:numFiles
    imgName=erase(files(i).name,'.xml');
%     if strcmp(imgName,'UCSF-OS; K34')==0
%         continue
%     end
    if ismember(imgName,onlyList)==false
          continue
    end
%      imgName
%    try
        outFile=[outFolder imgName '.png'];
        %if exist(outFile,'file')~=2 
            info=imfinfo([imgFolder imgName '.tif']);
            %tissueMask=double(imfill(imread([tissueMaskFolder imgName tissueMaskExt]),'holes'));
            annot=getAnnotation_ASAPformat([annotFolder files(i).name]);
            
            [~,ind] = max(cat(1,info.Height));
            %[h,w,~]=size(tissueMask);
            %M=buildMaskFromPoly(annot,info(ind).Height/h,h,w);
            M=buildMaskFromPoly(annot,16,info(ind).Height/16,info(ind).Width/16);
            
            if ismember(imgName,negativeList)
                M=tissueMask-M*255;
            end
            
            imwrite(M,outFile);
            %break
        %end
%    catch ex
%        fprintf('Error processing image %s: %s\n',imgName,ex.message);
%    end
end