function [cntR_total,sumR_total,cntP_total,sumP_total,scores,bestF] = prScores(supIm,iid,gt,nthresh,method,cntR_total,sumR_total,cntP_total,sumP_total,scores,iter,scorePath)

[thresh,cntR,sumR,cntP,sumP] = PR_core(supIm,gt,nthresh);
R = cntR  ./ (sumR  + (sumR ==0));
P  = cntP  ./ (sumP  + (sumP ==0));
[bestT,bestR,bestP,bestF] = maxF(thresh,R,P);
scores(iter,:) = [iid bestT bestR bestP bestF];

fname = fullfile(scorePath,[method '_scores.txt']);
fid = fopen(fname,'w');
if fid==-1,error('Could not open file %s for writing.',fname);end
fprintf(fid,'%10d %10g %10g %10g %10g\n',scores(1:iter,:)');
fclose(fid);

cntR_total = cntR_total  + cntR ;
sumR_total  = sumR_total  + sumR ;
cntP_total  = cntP_total  + cntP ;
sumP_total  = sumP_total  + sumP ;

R = cntR_total ./ (sumR_total + (sumR_total ==0));
P = cntP_total ./ (sumP_total + (sumP_total ==0));
F = fmeasure(R ,P);
[bestT,bestR,bestP,bestF] = maxF(thresh,R,P);
disp(['Image ID: ', num2str(iid), '  BestF: ', num2str(bestF)]);
fname = fullfile(scorePath,[method '_pr.txt']);
fid = fopen(fname,'w');
if fid==-1, error('Could not open file %s for writing.',fname); end
fprintf(fid,'%10g %10g %10g %10g\n',[thresh R P F]');
fclose(fid);
fname = fullfile(scorePath,[method '_score.txt']);
fid = fopen(fname,'w');
if fid==-1, error('Could not open file %s for writing.',fname); end
fprintf(fid,'%10g %10g %10g %10g\n',bestT,bestR,bestP,bestF);
fclose(fid);
