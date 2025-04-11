import os
import cv2
import spectral.io.envi as envi

# this class validates the image realibility and log the issues
class ImageValidator:
    def __init__(self, dataset_folder, is_hyperspectral=True, log_file='invalid_images.log'):
        self.dataset_folder = dataset_folder
        self.is_hyperspectral = is_hyperspectral
        self.log_file = log_file

    def lof_issues(self, message):
        '''logs the issues to a file for easy debugging'''
        with open(self.log_file, 'a') as log:
            log.write(message + '\n')
        print(message)
# check is all images are readable and logs errors if not
    def validate_image(self):
        image_files = sorted([img for img in os.listdir(self.dataset_folder) if
                              img.endswith('.bil') or img.endswith('.jpg') or img.endswith('.png')])
        hdr_files = sorted([img for img in os.listdir(self.dataset_folder) if img.endswith('.bil.hdr')])

        #quick check if file is found
        if not image_files:
            self.lof_issues('No image files found in the dataset folder!')
            return False

        print(f'Found {len(image_files)} image files to validate.')

        all_valid = True
        with open(self.log_file, 'w') as log:
            log.write('# Invalid images Log - Debugging missing or corrupt files\n\n')

        # loop throughout the image to see if everything is correct
        for image_file in image_files:
            image_path = os.path.join(self.dataset_folder, image_file)
            image_path = os.path.abspath(image_path)

            if self.is_hyperspectral:
                hdr_path = image_path + '.hdr'
                if not os.path.exists(hdr_path):
                    self.lof_issues(f'Missing HDR file for: {image_file}')
                    all_valid = False
                    continue

                # check quikcly why is not finding the hdr images
                try:
                    img = envi.open(hdr_path, image_path).load()
                    if img is None or img.shape[0] == 0:
                        self.log_issues(f'Failed to load hyprespectral image: {image_file}')
                        all_valid = False
                except Exception as err:
                        self.log_issues(f'Error loading hyprespectral image: {image_file}')
                        all_valid = False
            else:
                # if correct turn the image to grayscale
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.log_issues(f'Failed to load grayscale image: {image_file}')
                    all_valid = False
                else:
                    print(f'Image loaded successfully: {image_file}')
        print('Image validation completed! Errors are logged in invalid_images.log')
        return all_valid

# use it in the main execution file
if __name__ == '__main__':
    dataset_folder = "C:/Users/thega/OneDrive/Desktop/BSc Computer Science Year 3/CST3990 Undergraduate Individual Project/Project Folder/dataset-10610238/Extracted"
    validator = ImageValidator(dataset_folder, is_hyperspectral=True)

    print(f'Running image validation befpre training....')
    if not validator.validate_image():
        print("Validation failed! Check 'invalid_images.log' for details before proceeding.")
        exit()
    else:
        print("All images are valid. Proceeding to training...")