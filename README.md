# Digit Recognition using Python 


Built a prediction model for classifying and recognizing handwritten digits based on training a multi-class 
linear Support Vector Machine (SVM) classifier for predicting digits from the <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">MNIST</a> database of handwritten 
digits.(Python, OpenCV, Scikit-Learn , Scikit-Image and NumPy)



# Requirements

* In command line,navigate to the project folder and run the below command:

```

pip install --upgrade -r requirements.txt

```

This will install all the necessary requirements for running the application

* Clone the source - 

```
git clone git@github.com:saptarsi-chowdhury/handwritten_digit_recognition.git

```

* The next step is to train and generate the digit classifier. To do so run the script `digit_Classifier.py`. It will produce the classifier named `digits_classifier.pkl`. 

```
python digit_Classifier.py
```
* To test the classifier, run the `digits_Recognition.py` script.
```
python digits_Recognition.py -c <path of digit classifier file> -i <path of test image>
```
ex -
```
python digits_Recognition.py -c digits_classifier.pkl -i test_image_1.jpg
```

```
python digits_Recognition.py -c digits_classifier.pkl -i test_image_2.jpg
```


# Contents

This repository contains the following files-

1. `digit_Classifier.py` - Python Script to create the digits classifier file `digits_classifier.pkl`.

2. `digits_Recognition.py` - Python Script to test digits classifier for recognition of handwritten digits.

3. `digits_classifier.pkl` - Classifier file for digit recognition.

4. `test_image_1.jpg` - Test image number 1 to test the classifier

5. `test_image_2.jpg` - Test image numbre 2 to test the classifier

6. `predicted_image_1.jpg` - Resultant image with predicted digits from the image `test_image_1.jpg` 

7. `predicted_image_2.jpg` - Resultant image with predicted digits from the image `test_image_2.jpg` 

8. `final_image_1.jpg` - Resultant image with predicted digits compared with the original image `test_image_1.jpg`

9. `final_image_2.jpg` - Resultant image with predicted digits compared with the original image `test_image_2.jpg` 



# Results

### Image 1

![](/final_image_1.jpg)


### Image 2

![](/final_image_2.jpg)


