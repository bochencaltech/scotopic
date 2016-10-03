function [h b] = nhist(varargin)
    [h b] = hist(varargin{:});
    h = h / sum(h) / (b(2)-b(1));
end