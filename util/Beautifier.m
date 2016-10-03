% (c) 2015 Bo Chen 
% bchen3@caltech.edu
classdef Beautifier
    % utilities for plotting
    properties
        FS = 15; % fontsize
        MSdot = 30; % markersize for dots
        MSshape = 8; % markersize for shapes
        LW = 2;  % line width
    end
    methods
        function obj = papermode(obj)
            obj.LW = 3; obj.FS = 18; obj.MSdot = 40; obj.MSshape = 10;
        end
        function hdl = xlabel( obj, varargin )
           hdl = xlabel( varargin{:}, 'FontSize', obj.FS);  
        end
        
        function hdl = ylabel( obj, varargin )
           hdl = ylabel( varargin{:}, 'FontSize', obj.FS);  
        end
        
        function hdl = legend( obj, varargin )
           hdl = legend( varargin{:} );
           set(hdl, 'FontSize', obj.FS);  
        end
        
        function hdl = title( obj, varargin )
           hdl = title( varargin{:} );
           set(hdl, 'FontSize', obj.FS);  
        end
        
        function hld = errorbar(obj, varargin)
           % if line width not specified, set it
           if ~any(strcmp(varargin, 'LineWidth'))
                hdl = errorbar( varargin{:}, 'LineWidth', obj.LW ); 
           else
               hdl = errorbar( varargin{:} );
           end
           % if the marker type is ".", set it to FSdot ...
           marker = get(hdl, 'Marker');
           switch marker
               case 'none', % Do nothing
               case '.'
                    set(hdl, 'MarkerSize', obj.MSdot); 
               otherwise
                    set(hdl, 'MarkerSize', obj.MSshape);
           end
           p_hdl = get(hdl, 'Parent');
           if get(p_hdl, 'FontSize') ~= obj.FS;
               set(p_hdl, 'FontSize', obj.FS); 
           end
        end
        
        function hdl = plot( obj, varargin )
           % if line width not specified, set it
           if ~any(strcmp(varargin, 'LineWidth'))
                hdl = plot( varargin{:}, 'LineWidth', obj.LW ); 
           else
               hdl = plot( varargin{:} );
           end
           % if the marker type is ".", set it to FSdot ...
           marker = get(hdl, 'Marker');
           if ~any(strcmp(varargin, 'MarkerSize'))
               switch marker
                   case 'none', % Do nothing
                   case '.'
                        set(hdl, 'MarkerSize', obj.MSdot); 
                   otherwise
                        set(hdl, 'MarkerSize', obj.MSshape);
                        set(hdl, 'MarkerFaceColor', 'w');
               end
           end
           p_hdl = get(hdl, 'Parent');
           if get(p_hdl, 'FontSize') ~= obj.FS;
               set(p_hdl, 'FontSize', obj.FS); 
           end
        end

        function hdl = loglog( obj, varargin )
            hdl = plot( obj, varargin{:} );
            set(gca, 'XScale', 'log', 'YScale', 'log');   
        end
        
        function CLR = CLR( obj,  n )
           CLR = linspecer(n); 
        end
    end
end
