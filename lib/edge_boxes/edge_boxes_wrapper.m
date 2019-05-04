function all_boxes = edge_boxes_wrapper(image_filenames, output_filename)

%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;


%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = 0.6;%.65;     % step size of sliding window search
opts.beta  = 0.7;%.75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

%% process all images and detect Edge Box bounding box proposals (see edgeBoxes.m)
all_boxes = {};
for i=1:length(image_filenames)
    im = imread(image_filenames{i});
    tic, all_boxes{i} = edgeBoxes(im,model,opts); toc
end

%% convert the bounding boxes to the caffe input format (SelectiveSearch):
% ['ymin', 'xmin', 'ymax', 'xmax']
for i=1:length(image_filenames)
    all_boxes{i} = [all_boxes{i}(:,2) all_boxes{i}(:,1) all_boxes{i}(:,2)+all_boxes{i}(:,4) all_boxes{i}(:,1)+all_boxes{i}(:,3) all_boxes{i}(:,5)];
end

if nargin > 1
    all_boxes
    save(output_filename, 'all_boxes', '-v7');
end
