# nli-images
To run this project, please get all the following requirements:
1.  Download the pre-trained [word embedding](https://nlp.stanford.edu/projects/glove/) and [SNLI dataset](https://nlp.stanford.edu/projects/snli/) or [my drive](https://drive.google.com/a/um.edu.mt/file/d/0B40JtotizQfxZkZHWTZpRzh5Rmc/view?usp=sharing) and update data links in the *.config* files 
2. Download Flickr30k data set and features for each image (7x7x512). Or you can get it from my drive: [Flickr30k](https://drive.google.com/open?id=0B40JtotizQfxbFdLUHF6RmNpUXM),[file names](https://drive.google.com/open?id=0B40JtotizQfxMG81TVoteHlKdFU), [image features](https://drive.google.com/open?id=0B40JtotizQfxVjF3QWZXd1ZZUDg) then update the directories in the image_utils.py.
3. Run python main.py --config_file=bimpm_baseline.config

**view results:**
viewer.html is for viewing the prediction results. Since browsers do not allow html and js to access local file, you need to install a web server. I use a simple extension [Web server for Chrome](https://chrome.google.com/webstore/detail/web-server-for-chrome/ofhbbkphhbklhfoeikjpcbhemlocgigb?hl=en). Open this extension and set the folder to be where *.result* file locate. Update the [viewer.html](https://github.com/hoavt-54/nli-images/blob/master/viewer.html) file to view the desired result file.


**Prerequisite**: tensorflow 1.0.1, numpy, python 2 ...


**BiMPM**: https://github.com/zhiguowang/BiMPM

