function comp_results(out_dir, varargin)
%VIS_RESULTS Visualize detectron's training log
%
close all

% if ~exist(out_dir, 'dir')
%     mkdir(out_dir)
% end

numPlots = length(varargin)/2;
data = cell(numPlots,1);
tag = cell(numPlots,1);
APs = zeros(12,numPlots);

for ind = 1:numPlots
    ind_d = 2*ind-1;
    ind_t = 2*ind;
    data{ind} = load(varargin{ind_d});

    APs(:,ind)=[data{ind}.stats.AP, ...
                data{ind}.stats.AP50, ...
                data{ind}.stats.AP75, ...
                data{ind}.stats.APs, ...
                data{ind}.stats.APm, ...
                data{ind}.stats.APl, ...
                data{ind}.stats.AP50s, ...
                data{ind}.stats.AP50m, ...
                data{ind}.stats.AP50l, ...
                data{ind}.stats.AP75s, ...
                data{ind}.stats.AP75m, ...
                data{ind}.stats.AP75l];

    ARs(:,ind)=[data{ind}.stats.ARmd1, ...
                data{ind}.stats.ARmd10, ...
                data{ind}.stats.ARmd100, ...
                data{ind}.stats.ARs, ...
                data{ind}.stats.ARm, ...
                data{ind}.stats.ARl];

    tag{ind}= varargin{ind_t};

end

% Plotting Average Precision
fig1 = figure('visible','off');
x = categorical({'AP','AP50','AP75', 'APs', 'APm', 'APl', 'AP50s', 'AP50m', 'AP50l', 'AP75s', 'AP75m', 'AP75l'});
x = reordercats(x,{'AP','AP50','AP75', 'APs', 'APm', 'APl', 'AP50s', 'AP50m', 'AP50l', 'AP75s', 'AP75m', 'AP75l'});
bar(x, APs, 'grouped');
leg1=legend(tag, 'location','north');
set(leg1,'Interpreter', 'none');
% ylim([0,1])
title('Average Precision')
filename1 = [tag{1} '_VS_' tag{2} '_AP.jpg'];
file1 = fullfile(out_dir, filename1);
saveas(fig1, file1);

% Plotting Average Recall
fig2 = figure('visible','off');
x = categorical({'ARmd1','ARmd10','ARmd100', 'ARs', 'ARm', 'ARl'});
x = reordercats(x,{'ARmd1','ARmd10','ARmd100', 'ARs', 'ARm', 'ARl'});
bar(x, ARs, 'grouped');
leg2=legend(tag, 'location','north');
set(leg2,'Interpreter', 'none');
% ylim([0,1])
title('Average Recall')
filename2 = [tag{1} '_VS_' tag{2} '_AR.jpg'];
file2 = fullfile(out_dir, filename2);
saveas(fig2, file2);


disp('Matlab function finished sucessfully.')

end
