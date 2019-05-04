function vis_results(mat_file, out_dir)
%VIS_RESULTS Visualize detectron's training log
%


if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end

show_figs = 'off';
load(mat_file);

% Plot training loss
if isfield(stats, 'iter')
    fig1 = figure('visible',show_figs);
    fig1 = plot(stats.iter, stats.loss);
    ylim([0,0.1])
    title('Training loss vs. iteration')
    file1 = fullfile(out_dir, 'tr_loss.jpg');
    saveas(fig1, file1);

    % Plot training accuracy
    fig2 = figure('visible',show_figs);
    fig2 = plot(stats.iter, stats.accuracy_cls);
    title('Training class accuracy vs. Iteration')
    ylim([0.9,1])
    file2 = fullfile(out_dir, 'tr_acc.jpg');
    saveas(fig2, file2);

    % Plot bbox loss
    fig3 = figure('visible',show_figs);
    fig3 = plot(stats.iter, stats.loss_bbox);
    title('Training BBox Loss vs. Iteration')
    ylim([0,0.1])
    file3 = fullfile(out_dir, 'tr_loss_bbox.jpg');
    saveas(fig3, file3);

    % Plot bbox loss
    fig4 = figure('visible',show_figs);
    fig4 = plot(stats.iter, stats.loss_cls);
    title('Training Class Loss vs. Iteration')
    ylim([0,0.1])
    file4 = fullfile(out_dir, 'tr_loss_class.jpg');
    saveas(fig4, file4);
end

% Plot Validation Average Precision
fig5 = figure('visible',show_figs);
y = [stats.AP; stats.AP50; stats.AP75; stats.APs; stats.APm; stats.APl; stats.AP50s; stats.AP50m; stats.AP50l; stats.AP75s; stats.AP75m; stats.AP75l];
x = categorical({'AP','AP50','AP75', 'APs', 'APm', 'APl', 'AP50s', 'AP50m', 'AP50l', 'AP75s', 'AP75m', 'AP75l'});
x = reordercats(x,{'AP','AP50','AP75', 'APs', 'APm', 'APl', 'AP50s', 'AP50m', 'AP50l', 'AP75s', 'AP75m', 'AP75l'});
fig5 = bar(x,y);
% ylim([0,1])
text(1:length(y),y,num2str(y),'vert','bottom','horiz','center');
box off
title('Average Precision')
file5 = fullfile(out_dir, 'val_AP.jpg');
saveas(fig5, file5);

% Plot Validation Average Recall
fig6 = figure('visible',show_figs);
y = [stats.ARmd1; stats.ARmd10; stats.ARmd100; stats.ARs; stats.ARm; stats.ARl];
x = categorical({'ARmd1','ARmd10','ARmd100', 'ARs', 'ARm', 'ARl'});
x = reordercats(x,{'ARmd1','ARmd10','ARmd100', 'ARs', 'ARm', 'ARl'});
fig6 = bar(x,y);
% ylim([0,1])
text(1:length(y),y,num2str(y),'vert','bottom','horiz','center');
box off
title('Average Recall')
file6 = fullfile(out_dir, 'val_AR.jpg');
saveas(fig6, file6);

disp('Matlab finished successfully.')
