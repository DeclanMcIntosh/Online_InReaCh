import cv2
import numpy as np
import os
import random
from typing import List, Tuple

def resize_and_crop(image: np.ndarray, size: Tuple[int, int], crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize and crop an image to the specified size and crop dimensions.
    
    Args:
        image (np.ndarray): The input image.
        size (Tuple[int, int]): The target size for resizing (width, height).
        crop_size (Tuple[int, int]): The target size for cropping (width, height).
    
    Returns:
        np.ndarray: The resized and cropped image.
    """
    resized = cv2.resize(image, size)
    x = (size[0] - crop_size[0]) // 2
    y = (size[1] - crop_size[1]) // 2
    # Resize image to have 3 channels if it has only 1 channel
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    elif resized.shape[2] == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    # Ensure the image has 3 channels
    if resized.shape[2] != 3:
        raise ValueError("Image must have 3 channels after resizing.")

    return resized[y:y + crop_size[1], x:x + crop_size[0]]

def load_corrupted_data(class_name: str, 
                        data_dir: str, 
                        num_corrupted: int, 
                        num_nominal: int = 99999999, 
                        size: Tuple[int, int] = (256, 256), 
                        crop_size: Tuple[int, int] = (224, 224),
                        worst_case: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load corrupted and nominal data for a given class.

    Args:
        class_name (str): The class name to load data for.
        data_dir (str): The root directory of the dataset.
        num_corrupted (int): Number of corrupted samples to load.
        num_nominal (int): Number of nominal samples to load.
        size (Tuple[int, int]): Resize dimensions (width, height).
        crop_size (Tuple[int, int]): Crop dimensions (width, height).
        worst_case (bool): If True, do not shuffle the data.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple of images and masks.
    """
    train_images = load_training_data(class_name, data_dir, size=size, crop_size=crop_size)
    test_images, test_masks, _ = load_testing_data(class_name, data_dir, size=size, crop_size=crop_size)
    
    images = test_images[:num_corrupted] + train_images[:num_nominal]
    masks = test_masks[:num_corrupted] + [np.zeros_like(test_masks[0]) for _ in range(len(train_images))][:num_nominal]
    
    if not worst_case:
        combined = list(zip(images, masks))
        random.shuffle(combined)
        images, masks = zip(*combined)
    
    return list(images), list(masks)

def load_testing_data(class_name: str, 
                      data_dir: str, 
                      size: Tuple[int, int] = (256, 256), 
                      crop_size: Tuple[int, int] = (224, 224)) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load testing data for a given class.

    Args:
        class_name (str): The class name to load data for.
        data_dir (str): The root directory of the dataset.
        size (Tuple[int, int]): Resize dimensions (width, height).
        crop_size (Tuple[int, int]): Crop dimensions (width, height).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[str]]: A tuple of test images, ground truth masks, and class labels.
    """
    img_dir = os.path.join(data_dir, class_name, 'test')
    ann_dir = os.path.join(data_dir, class_name, 'ground_truth')
    test_images, test_truths, test_classes = [], [], []

    for directory in os.listdir(img_dir):
        dir_path = os.path.join(img_dir, directory)
        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            image = cv2.imread(img_path)
            test_images.append(resize_and_crop(image, size, crop_size))
            
            if directory != 'good':
                mask_path = os.path.join(ann_dir, directory, f"{filename[:-4]}_mask.png")
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                test_truths.append(resize_and_crop(mask, size, crop_size))
            else:
                test_truths.append(np.zeros_like(test_images[-1]))
            
            test_classes.append(directory)

    combined = list(zip(test_images, test_truths, test_classes))
    random.shuffle(combined)
    test_images, test_truths, test_classes = zip(*combined)

    return list(test_images), list(test_truths), list(test_classes)

def load_training_data(class_name: str, 
                       data_dir: str, 
                       size: Tuple[int, int] = (224, 224), 
                       crop_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """
    Load training data for a given class.

    Args:
        class_name (str): The class name to load data for.
        data_dir (str): The root directory of the dataset.
        size (Tuple[int, int]): Resize dimensions (width, height).
        crop_size (Tuple[int, int]): Crop dimensions (width, height).

    Returns:
        List[np.ndarray]: A list of training images.
    """
    dir_path = os.path.join(data_dir, class_name, 'train', 'good')
    train_images = []

    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)
        image = cv2.imread(img_path)
        train_images.append(resize_and_crop(image, size, crop_size))
    
    return train_images


if __name__ == "__main__":
    import tempfile
    import shutil

    def create_mock_dataset(base_dir: str):
        """
        Create a mock dataset structure with synthetic images and masks for testing.
        """
        os.makedirs(os.path.join(base_dir, "class1", "train", "good"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "class1", "test", "good"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "class1", "test", "defect1"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "class1", "ground_truth", "defect1"), exist_ok=True)

        # Create synthetic training images
        for i in range(5):
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(base_dir, "class1", "train", "good", f"train_{i}.png"), img)

        # Create synthetic testing images and masks
        for i in range(3):
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            mask = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            cv2.imwrite(os.path.join(base_dir, "class1", "test", "defect1", f"test_{i}.png"), img)
            cv2.imwrite(os.path.join(base_dir, "class1", "ground_truth", "defect1", f"test_{i}_mask.png"), mask)

        # Create synthetic "good" testing images
        for i in range(2):
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(base_dir, "class1", "test", "good", f"good_{i}.png"), img)

    # Create a temporary directory for the mock dataset
    temp_dir = tempfile.mkdtemp()
    try:
        create_mock_dataset(temp_dir)

        # Test `load_training_data`
        print("Testing `load_training_data`...")
        train_images = load_training_data("class1", temp_dir)
        print(f"Loaded {len(train_images)} training images.")
        assert len(train_images) == 5, "Training data loading failed."
        for img in train_images:
            assert img.shape[:2] == (224, 224), f"Training image size mismatch: {img.shape[:2]}"

        # Test `load_testing_data`
        print("Testing `load_testing_data`...")
        test_images, test_masks, test_classes = load_testing_data("class1", temp_dir)
        print(f"Loaded {len(test_images)} testing images and {len(test_masks)} masks.")
        assert len(test_images) == 5, "Testing data loading failed."
        assert len(test_masks) == 5, "Testing masks loading failed."
        for img, mask in zip(test_images, test_masks):
            assert img.shape[:2] == (224, 224), f"Testing image size mismatch: {img.shape[:2]}"
            assert mask.shape[:2] == (224, 224), f"Testing mask size mismatch: {mask.shape[:2]}"

        # Test `load_corrupted_data`
        print("Testing `load_corrupted_data`...")
        images, masks = load_corrupted_data("class1", temp_dir, num_corrupted=3, num_nominal=2)
        print(f"Loaded {len(images)} images and {len(masks)} masks.")
        assert len(images) == 5, "Corrupted data loading failed."
        assert len(masks) == 5, "Corrupted masks loading failed."
        for img, mask in zip(images, masks):
            assert img.shape[:2] == (224, 224), f"Corrupted image size mismatch: {img.shape[:2]}"
            assert mask.shape[:2] == (224, 224), f"Corrupted mask size mismatch: {mask.shape[:2]}"

        print("All tests passed!")

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)