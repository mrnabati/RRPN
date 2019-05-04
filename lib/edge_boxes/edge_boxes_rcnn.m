function all_boxes = edge_boxes_rcnn(image_filenames, output_filename)
    addpath(genpath('./toolbox/'));

    %% load pre-trained edge detection model and set opts (see edgesDemo.m)
    model=load('models/forest/modelBsds'); model=model.model;
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    
    
    %% set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = 0.6;%.65;     % step size of sliding window search
    opts.beta  = 0.7;%.75;     % nms threshold for object proposals
    opts.minScore = .02;  % min score of boxes to detect
    opts.maxBoxes = 1e4;  % max number of boxes to detect
    
    %% process all images and detect Edge Box bounding box proposals (see edgeBoxes.m)
    % Prepare the image filenames
    names = fileread(image_filenames);
    split_names = strsplit(names);
    num_imgs = length(split_names);
    
    all_boxes = {};
    tic;
    for i=1:num_imgs
        filename = split_names{i};
        if isempty(filename)
            continue;
        end
        im = imread(filename);
        all_boxes{i} = edgeBoxes(im,model,opts);
        fprintf('image %d/%d, No. boxes: %d\n', i, num_imgs,length(all_boxes{i}))
    end
    toc;
    
    %% convert the bounding boxes to the caffe input format (SelectiveSearch):
    % ['ymin', 'xmin', 'ymax', 'xmax', 'score']
    num_imgs = length(all_boxes);
    for i=1:num_imgs
        all_boxes{i} = [all_boxes{i}(:,2) all_boxes{i}(:,1) all_boxes{i}(:,2)+all_boxes{i}(:,4) all_boxes{i}(:,1)+all_boxes{i}(:,3)];
    end
    
    % Save results to mat file
    boxes = all_boxes;
    save(output_filename, 'boxes', '-v7.3');

    statement1 = ['Edge Boxes proposals saved to: ' output_filename];
    statement2 = ['No. Images: ' num2str(num_imgs) ',   Total Time: ' num2str(toc) 's,  Time/Image: ' num2str(toc/num_imgs) 's'];
    disp(statement1)
    disp(statement2)
