function all_boxes = selective_search(image_filenames, output_filename)

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

colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation.
minSize = k;
sigma = 0.8;

% Process all images.
all_boxes = {};
names = fileread(image_filenames);
% str = input('say something...','s')

split_names = strsplit(names);
num_imgs = length(split_names);
tic;
for i=1:num_imgs
    filename = split_names{i};
    if isempty(filename)
      continue;
    end
    im = imread(filename);
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    all_boxes{i} = BoxRemoveDuplicates(boxes);
end
toc;
% Save results to mat file
save(output_filename, 'all_boxes', '-v7.3');


statement1 = ['Selective Search proposals saved to: ' output_filename];
statement2 = ['No. Images: ' num2str(num_imgs) ',   Total Time: ' num2str(toc)];
disp(statement1)
disp(statement2)
str = input('say something...','s')
