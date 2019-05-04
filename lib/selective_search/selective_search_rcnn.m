function all_boxes = selective_search_rcnn(image_filenames, output_filename, fast_mode, im_width)

% Based on the demo.m file included in the Selective Search
% IJCV code, and on selective_search_boxes.m from R-CNN.

% Load dependencies and compile if needed.

addpath('Dependencies');

if(~exist('anigauss'))
    mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
end

if(~exist('mexCountWordsIndex'))
    mex Dependencies/mexCountWordsIndex.cpp
end

if(~exist('mexFelzenSegmentIndex'))
    mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, ...
                      @SSSimTextureSizeFill, ...
                      @SSSimBoxFillOrig, ...
                      @SSSimSize};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
% controls size of segments of initial segmentation.
ks = [50 100 150 300];
sigma = 0.8;

% After segmentation, filter out boxes which have a width/height smaller
% than minBoxWidth (default = 20 pixels).
minBoxWidth = 20;

% Comment the following three lines for the 'quality' version
if fast_mode
  disp('Fast mode\n')
  colorTypes = colorTypes(1:2); % 'Fast' uses HSV and Lab
  simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
  ks = ks(1:2);
end

% Prepare the image filenames
names = fileread(image_filenames);
split_names = strsplit(names);
num_imgs = length(split_names);
% Process all images.
all_boxes = {};
tic;
for i=1:num_imgs
    fprintf('image %d/%d\n', i, num_imgs)
    filename = split_names{i};
    if isempty(filename)
      continue;
    end
    im = imread(filename);

    % Resize image to canonical dimensions since proposals aren't scale invariant.
    scale = size(im, 2) / im_width;
    im = imresize(im, [NaN im_width]);

    idx = 1;
    for j = 1:length(ks)
      k = ks(j); % Segmentation threshold k
      minSize = k; % We set minSize = k
      for n = 1:length(colorTypes)
        colorType = colorTypes{n};
        [boxesT{idx} blobIndIm blobBoxes hierarchy priorityT{idx}] = ...
          Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
        idx = idx + 1;
      end
    end
    boxes = cat(1, boxesT{:}); % Concatenate boxes from all hierarchies
    priority = cat(1, priorityT{:}); % Concatenate priorities

    % Do pseudo random sorting as in paper
    priority = priority .* rand(size(priority));
    [priority sortIds] = sort(priority, 'ascend');
    boxes = boxes(sortIds,:);

    boxes = FilterBoxesWidth(boxes, minBoxWidth);
    boxes = BoxRemoveDuplicates(boxes);

    % Adjust boxes to cancel effect of canonical scaling.
    boxes = (boxes - 1) * scale + 1;

    boxes = FilterBoxesWidth(boxes, minBoxWidth);
    boxes = BoxRemoveDuplicates(boxes);
    all_boxes{i} = boxes;
end
toc;

% Save results to mat file
boxes = all_boxes;
save(output_filename, 'boxes', '-v7.3');


statement1 = ['Selective Search proposals saved to: ' output_filename];
statement2 = ['No. Images: ' num2str(num_imgs) ',   Total Time: ' num2str(toc) 's,  Time/Image: ' num2str(toc/num_imgs) 's'];
disp(statement1)
disp(statement2)
