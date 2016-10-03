function printplot(OUTFILE, printcode)
% printplot(OUTFILE, printcode)
% print the current plot into pdf specified by OUTFILE
% OUTFILE: name of output file without ".pdf" at the end
% printcode: where to put the current plot
%   0: start a new file
%   1: append to file
%   2: finish and convert file into pdf
%   3: do not append, just finish and convert file to pdf
% (c) Bo Chen
% bchen3@caltech.edu

    if ~exist('printcode', 'var'), printcode = 1; end
    orient landscape;
    if length(OUTFILE)<3 || ~strcmp(OUTFILE(end-2:end),'.ps'),
        OUTFILE_PS = [OUTFILE '.ps'];
    else
        OUTFILE_PS = OUTFILE;
    end
    if printcode==0,
        print('-dpsc',OUTFILE_PS);
    elseif printcode <= 2
        print('-dpsc','-append',OUTFILE_PS);
    end
    if printcode >= 2,
        system(sprintf('ps2pdf %s.ps %s.pdf',OUTFILE, OUTFILE));
        system(sprintf('rm %s.ps', OUTFILE));
        id = strfind(OUTFILE, 'public_html');
        if ~isempty(id)
            fprintf(['Plots saved to:\n'...
            'http://vision.caltech.edu/~bchen3/%s.pdf\n'], ...
                OUTFILE(id+12:end));
        else
            fprintf('Plots saved to: %s.pdf\n', OUTFILE);
        end
    end
        
end