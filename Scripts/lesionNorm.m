function lesionNorm (finalChar)
if ~exist('finalChar','var')
    finalChar = ''
     %for parallel e.g. use '1' to only process files which end with 1
end
inDir = pwd;
outDir = '/Users/chris/norm';
tempDir = ['/Users/chris/temp', finalChar];
ext = '.nii.gz';
pth = fullfile(inDir,['*_lesion', ext]);
lesions = dirVisible(pth);
if length(lesions) < 1 
    error("Is `%s` the correct file extension? Unable to find lesions %s\n", ext, pth)
end
nOK = 0;
for i = 1: length(lesions)
    lesion = fullfile(lesions(i).folder, lesions(i).name);
    parts = split(lesions(i).name,'_');
    subj = parts{1};
    if length(finalChar) > 0
        if ~endsWith(subj, finalChar)
            fprintf('Final Character does not match: skipping %s\n', subj);
            continue;
        end
    end
    T1 = fullfile(lesions(i).folder, strrep(lesions(i).name, '_lesion', '_T1w'));
    if ~exist(T1, 'file')
        error('Unable to find %s\n', T1)
    end
    T2 = fullfile(lesions(i).folder, strrep(lesions(i).name, '_lesion', '_T2w'));
    if ~exist(T2, 'file')
        error('Unable to find %s\n', T2)
    end
    delete(fullfile(tempDir, '*'));
    les = gunzip(lesion, tempDir);
    les = les{1};
    [p,n,x] = fileparts(les);
    outLes = fullfile(outDir, ['w',n,x]);
    if (exist(outLes))
        fprintf('Skipping existing lesion %s\n', outLes)
        continue;
    end
    
    T1w = gunzip(T1, tempDir);
    T1w = T1w{1};
    T2w = gunzip(T2, tempDir);
    T2w = T2w{1};
    clinical_setorigin(strvcat(T1w),1);
    clinical_setorigin(strvcat(T2w,les),2);
    clinical_mrnormseg12(T1w,les,T2w);
    copyfile(fullfile(tempDir, ['wsr',n,x]), outLes)
    [p,n,x] = fileparts(T1w);
    copyfile(fullfile(tempDir, ['wb',n,x]), fullfile(outDir, ['wb',n,x]))
    copyfile(fullfile(tempDir, ['e',n,x]), fullfile(outDir, ['e',n,x]))
    nOK = nOK + 1;
    fprintf('Processed %d/%d files\n', nOK, length(lesions));
end
fprintf('Processed %d files\n', nOK);

function d=dirVisible(pathFolder)
d = dir(pathFolder);
d = d(arrayfun(@(d) ~strcmp(d.name(1),'.'),d));
%end dirVisible()