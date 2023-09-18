# FaceMorph
Simple image morphing operation using OpenCV; copied, and pasted into one unified script for ease of operations

## Installation
Clone the repository to your local machine:
```shell
git clone https://github.com/razyoboy/FaceMorph.git
```

Install the required Python packages:
```shell
pip install -r requirements.txt
```

## Usage
To use FaceMorph, you'll need to navigate to the src directory within the project folder. Use the following command to change to the correct directory:

```shell
cd src
```
[!NOTE]\
The project was developed using PyCharm, which has specific path configurations. It is essential to run the script from the `src` directory to avoid path-related issues.

### Modes of Operation
The main.py script can operate in two modes: batch and non-batch. Here's how you can configure each mode:

#### Non-Batch Mode
In non-batch mode, you will specify two individual images to morph together. To use this mode, set the `BATCH_MORPH` variable to False and update the `filename1` and `filename2` variables with the paths to your chosen images.

```python
BATCH_MORPH = False
filename1 = "path/to/your/image1.jpg"
filename2 = "path/to/your/image2.jpg"
```
#### Batch Mode
In batch mode, the script will morph together every possible pair of images in a specified folder, utilizing the combination formula to ensure each image has a unique set of outcomes. To use this mode, set the `BATCH_MORPH` variable to True and update the `folder_path` variable with the path to your folder containing the images.
```python
BATCH_MORPH = True
folder_path = 'path/to/your/folder'
```

Then you can run the script using:
```sh
python main.py
```
