This is mostly Piotr Dollar's code for Edge Boxes object proposals from [Edge Boxes: Locating Object Proposals from Edges](https://github.com/pdollar/edges), downloaded in July 2015.
I needed a way to call this stuff from Python; `edge_boxes.py` and `edge_boxes_wrapper.m` are the only new files you need to use the code. They have been adapted from 
Sergey Karayev's [Selective Search IJCV with Python](https://github.com/sergeyk/selective_search_ijcv_with_python) code, which wraps an alternative object-proposal generator.

Make sure that the edges directory is in your PYTHONPATH and just do:

	import edges
	
	windows = edge_boxes.get_windows(image_filenames)

To make sure this works, simply `python edge_boxes.py`.

Finally, I needed the code to be able to use the proposals together with Ross Girshick's Fast R-CNN: [Fast Region-based Convolutional Networks for object detection](https://github.com/rbgirshick/fast-rcnn) .
A demo file for this is also included (`demo_edgeboxes.py`). To try it out build Fast R-CNN and drop the file in its 'tools' subdirectory. 

The license is the same as for Piotr Dollar's Structured Edge Detection Toolbox V3.0 (see `license.txt`) and his original readme is in `sedt_readme.txt`.

Enjoy!

Dubravko Culibrk
22 Jul 2015

P.S. Please note that the code uses Piotr's MATLAB toolbox (https://pdollar.github.io/toolbox/), which needs to be installed for any of this to work. (Thanks to Thomas Lau for pointing out that this should be stated in this README).
