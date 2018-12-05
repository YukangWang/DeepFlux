function evaluation(detDir, gtDir)
% skeleton detection evaluation

addpath(genpath('Util'));
addpath(genpath('nonmax'));

method = 'DeepFlux';
nthresh = 10;
scorePath = 'results';
if ~exist(scorePath, 'dir'), mkdir(scorePath); end

items = dir([gtDir, '*.mat']);
items = {items.name};
gts = cell(1,length(items));
dets = cell(1,length(items));

for i=1:length(items)
    [~, fn, ~] = fileparts(items{i});
    gt = load([gtDir, fn, '.mat']);
    if ismember('symmetry',fieldnames(gt)), gt = logical(gt.symmetry);
    elseif ismember('sym',fieldnames(gt)), gt = logical(gt.sym); % for wh-symmax
    elseif ismember('gt',fieldnames(gt)), gt = logical(gt.gt); % for symmax300
    else assert(false);
    end

    det = single(imread([detDir, fn, '.png']));
    det = det / max(det(:));
    gts{i} = gt;
    dets{i} = det;
end

assert(iscell(gts));
assert(iscell(dets));
assert(length(gts) == length(dets));

cntR_color = zeros(nthresh,1);
sumR_color = zeros(nthresh,1);
cntP_color = zeros(nthresh,1);
sumP_color = zeros(nthresh,1);
scores_color = zeros(length(gts),5);

for i=1:length(gts)
   gt = gts{i};  det = dets{i};
   assert(islogical(gt));
   assert(sum(abs(size(gt) - size(det))) == 0); % check if size(gt)==size(det)
   det = nms(det);
   [cntR_color,sumR_color,cntP_color,sumP_color,scores_color,best_f] = ...
    prScores(det,i,gt,nthresh,method,cntR_color,sumR_color,cntP_color,sumP_color,scores_color,i,scorePath);
end

disp(['-----F-measure=', num2str(best_f), '-----']);
%fname = [scorePath, method, '_results.txt'];
%fid = fopen(fname,'a');
%fprintf(fid,'%10g %10g %10g %10g\n', best_f);
%fclose(fid);

end
