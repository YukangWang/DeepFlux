function [thresh,cntR,sumR,cntP,sumP] = symmetryPR(pb,skels,nthresh)
% function [thresh,cntR,sumR,cntP,sumP] = symmetryPR(pb,skels,nthresh)
%
% Calculate precision/recall curve.
% If pb is binary, then a single point is computed.
% The pb image can be smaller than the segmentations.
%
% INPUT
%	pb		Soft or hard symmetry map.
%	skels		Array of skeletons for different segmentations.
%	[nthresh]	Number of points in PR curve.
%
% OUTPUT
%	thresh		Vector of threshold values.
%	cntR,sumR	Ratio gives recall.
%	cntP,sumP	Ratio gives precision.
%
% See also boundaryPRfast.
% 
% David Martin <dmartin@eecs.berkeley.edu>
% January 2003

if nargin<3, nthresh = 100; end
if islogical(pb), 
    nthresh = 1;
    pb = double(pb);
end
nthresh = max(1,nthresh);
thresh = linspace(1/(nthresh+1),1-1/(nthresh+1),nthresh)';

% make sure the boundary maps are thinned to a standard thickness
nskels = size(skels,3);
for i = 1:nskels,
  skels(:,:,i) = skels(:,:,i) .* bwmorph(skels(:,:,i),'thin',inf);
end
skels = double(skels);

% zero all counts
cntR = zeros(size(thresh));
sumR = zeros(size(thresh));
cntP = zeros(size(thresh));
sumP = zeros(size(thresh));

if nthresh>1, progbar(0,nthresh); end
for t = 1:nthresh,
  % threshold pb to get binary boundary map
  bmap = (pb>=thresh(t));
  % thin the thresholded pb to make sure boundaries are standard thickness
  bmap = bwmorph(bmap,'thin',Inf);
  bmap = bwmorph(bmap,'clean');
  bmap = double(bmap);  
  % accumulate machine matches, since the machine pixels are
  % allowed to match with any segmentation
  accP = zeros(size(pb));
  % compare to each seg in turn
  for i = 1:nskels,
    %fwrite(2,'+');
    % compute the correspondence
    [match1,match2] = correspondPixels(bmap,skels(:,:,i),0.01);
    % accumulate machine matches
    accP = accP | match1;
    % compute recall
    sumR(t) = sumR(t) + sum(sum(skels(:,:,i)));
    cntR(t) = cntR(t) + sum(match2(:)>0);
  end
  % compute precision
  sumP(t) = sumP(t) + sum(bmap(:));
  cntP(t) = cntP(t) + sum(accP(:));
  if nthresh>1, progbar(t,nthresh); end
end
