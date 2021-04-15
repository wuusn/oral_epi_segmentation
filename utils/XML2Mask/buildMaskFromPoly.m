function M = buildMaskFromPoly(annot,scale,h,w)
%BUILDMASKFROMPOLY Summary of this function goes here
%   Detailed explanation goes here

M=zeros(h,w);
numReg=length(annot);
for j=1:numReg
    if(isempty(annot(j).regions))
        continue
    end
    x=annot(j).regions.X/scale;
    y=annot(j).regions.Y/scale;
    M=M+poly2mask(x,y,h,w);
end

end

