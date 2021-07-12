function SymOcLgHSV = dirSymMT(inputFile)
    warning off;
    restoredefaultpath;
    addpath(genpath(fullfile('.','libs')));
    % tic;
    [SymOcLgHSV,voteMap] = symBilOurCentLogGaborHSV(inputFile);
    % toc;
end