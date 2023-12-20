function bidsHarvest
bidsDir = '/Volumes/4tb/bids/';
outDir = pwd;
ext = '.nii.gz';
pth = fullfile(bidsDir, '**',['*lesion_mask', ext]);
lesions = dirVisible(pth);
if length(lesions) < 1 
    error("Is `%s` the correct file extension? Unable to find lesions %s\n", ext, pth)
end
nOK = 0;
for i = 1: length(lesions)
    lesion = fullfile(lesions(i).folder, lesions(i).name);
    if checkUnderscoreInPath(lesion)
        fprintf("Underscore, so skipping %s_%s\n",lesion);
        continue;
    end
    %parse subj and session sub-M2001_ses-1253_acq-spc3_run-3_T2w_desc-lesion_mask    
    fnm = lesions(i).name;
    parts = split(fnm,'_');
    subj = parts{1};
    sess = parts{2};
    outNm = fullfile(outDir, [subj,'_',sess,'*', ext]);
    if ~isempty(dir(outNm))
        %fprintf("Exists, so skipping %s_%s\n", subj, sess);
        continue;
    end
    fprintf("Harvesting %s\n", fnm)
    T2nm = split(fnm,'_desc');
    T2nm = T2nm{1};
    T2nm = fullfile(bidsDir, subj, sess, '**',[T2nm, ext]);
    T2s = dirVisible(T2nm);
    if length(T2s) < 1 
        error("Unable to find %s\n", T2nm)
    end
    T2 = fullfile(T2s(1).folder, T2s(1).name);
    T1nm = fullfile(T2s(1).folder,['*T1w', ext]);
    T1s = dirVisible(T1nm);
    bestDateT1 = 0;
    if length(T1s) < 1  
        parts = split(sess,'-');
        sessDate = str2num(parts{2});
        allses = dirVisible(fullfile(bidsDir, subj));
        bestDateError = Inf;
        for j = 1: length(allses)
            %folder names hold session "ses-1253"
            fnmj = allses(j).name;
            
            parts = split(fnmj,'-');
            sessjDate = str2num(parts{2});
            dateError = abs(sessDate - sessjDate);
            if (dateError < bestDateError)
                %check if there is a T1 scan in this folder
                T1nmj = fullfile(allses(1).folder, allses(j).name, 'anat', ['*T1w', ext]);
                fprintf("%s\n", T1nmj);
                T1sj = dirVisible(T1nmj);
                if length(T1sj) > 0
                    T1nm = T1nmj;
                    T1s = T1sj;
                    bestDateError = dateError;
                    bestDateT1 = sessjDate; 
                    %fprintf("Found T1 %d days from T2\n", bestDateError);
                end
            end
            %fprintf("%s: %d - %d = %d %g\n", subj, sessDate, sessjDate, dateError, bestDateError);
        end
        if length(T1s) < 1
            fprintf("Unable to find %s\n", T1nm);
            continue;
        else 
            fprintf("Found a T1 scan %d days from T2: %s\n", bestDateError, T1nm);
        end
    end
    T1 = fullfile(T1s(1).folder, T1s(1).name);
    sessX = sess;
    if bestDateT1 > 0
        %show days between T1 and T2
        sessX = [sess, 'x', num2str(bestDateT1)];
    end
    %fprintf("%s %s %s\n", lesion, T2, T1);
    copyfile(lesion, fullfile(outDir, [subj,'_',sessX, '_lesion', ext]))
    copyfile(T2, fullfile(outDir, [subj,'_',sessX, '_T2w', ext]))
    copyfile(T1, fullfile(outDir, [subj,'_',sessX, '_T1w', ext]))
    nOK = nOK + 1;  
end
fprintf("Harvested %d images\n", nOK)
%end bidsHarvest

function hasUnderscore = checkUnderscoreInPath(filePath)
% Split the file path into individual folder names
pathParts = strsplit(filePath, filesep);
% Check if any folder or the file name starts with a underscore
hasUnderscore = any(cellfun(@(part) startsWith(part, '_'), pathParts));
%checkUnderscoreInPath()

function d=dirVisible(pathFolder)
d = dir(pathFolder);
d = d(arrayfun(@(d) ~strcmp(d.name(1),'.'),d));
%we use underscores for bad files
d = d(arrayfun(@(d) ~strcmp(d.name(1),'_'),d));
%end dirVisible()