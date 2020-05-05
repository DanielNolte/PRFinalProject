classdef wormGUI < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure         matlab.ui.Figure
        ImageAxes        matlab.ui.control.UIAxes
        WormButton       matlab.ui.control.Button
        NowormButton     matlab.ui.control.Button
        MultiWormButton  matlab.ui.control.Button
        UnknownButton    matlab.ui.control.Button
    end
    
    
    methods (Access = public)
        
        function updateimage(app,imagefile)
           global busy
           busy = true;
           global im
           im = imagefile;
            try
                imshow(imagefile, 'parent', app.ImageAxes);
            catch ME
                % If problem reading image, display error message
                uialert(app.UIFigure, ME.message, 'Image Error');
                return;
            end
        end
    end
    

    % Callbacks that handle component events
    methods (Access = public)

        % Code that executes after component creation
        function startupFcn(app)
            % Configure image axes
            app.ImageAxes.Visible = 'off';
            app.ImageAxes.Colormap = gray(256);
            axis(app.ImageAxes, 'image');
            
            % Update the image and histograms
            updateimage(app, 'peppers.png');
        end

        % Callback function
        function WormButtonPushed(app, event)
            global im
            global fullImageName
            global i
            i
            wormFolder = 'data/worm/'
            imwrite(im, strcat(wormFolder,fullImageName,'_',int2str(i),'.jpg'));
            pause(0.1)
            global busy
            busy = false;
            
        end
        function NowormButtonPushed(app, event)
            global im
            global fullImageName
            global i
            i
            nowormFolder = 'data/noworm/'
            imwrite(im, strcat(nowormFolder,fullImageName,'_',int2str(i),'.jpg'));
            pause(0.1)
            global busy
            busy = false;
        end
        function MultiWormButtonPushed(app, event)
            global im
            global fullImageName
            global i
            i
            MultiWormFolder = 'data/multiworm/'
            imwrite(im, strcat(MultiWormFolder,fullImageName,'_',int2str(i),'.jpg'));
            pause(0.1)
            global busy
            busy = false;
        end
        function UnknownButtonPushed(app, event)
            global im
            global fullImageName
            global i
            i
            unknownFolder = 'data/unknown/'
            imwrite(im, strcat(unknownFolder,fullImageName,'_',int2str(i),'.jpg'));
            pause(0.1)
            global busy
            busy = false;
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Position = [100 100 567 480];
            app.UIFigure.Name = 'wormGUI';
            app.UIFigure.Resize = 'off';

            % Create ImageAxes
            app.ImageAxes = uiaxes(app.UIFigure);
            app.ImageAxes.XTick = [];
            app.ImageAxes.XTickLabel = {'[ ]'};
            app.ImageAxes.YTick = [];
            app.ImageAxes.Position = [77 147 300 300];

            % Create WormButton
            app.WormButton = uibutton(app.UIFigure, 'push','ButtonPushedFcn', @(WormButton,event) WormButtonPushed(app,event));
            app.WormButton.Position = [55 63 100 22];
            app.WormButton.Text = 'Worm';

            % Create NowormButton
            app.NowormButton = uibutton(app.UIFigure, 'push','ButtonPushedFcn', @(NowormButton,event) NowormButtonPushed(app,event));
            app.NowormButton.Position = [188 63 100 22];
            app.NowormButton.Text = 'No worm';

            % Create MultiWormButton
            app.MultiWormButton = uibutton(app.UIFigure, 'push','ButtonPushedFcn', @(MultiWormButton,event) MultiWormButtonPushed(app,event));
            app.MultiWormButton.Position = [311 63 100 22];
            app.MultiWormButton.Text = 'Multi Worm';

            % Create UnknownButton
            app.UnknownButton = uibutton(app.UIFigure, 'push','ButtonPushedFcn', @(UnknownButton,event) UnknownButtonPushed(app,event));
            app.UnknownButton.Position = [436 63 100 22];
            app.UnknownButton.Text = 'Unknown/Both';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = wormGUI

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end