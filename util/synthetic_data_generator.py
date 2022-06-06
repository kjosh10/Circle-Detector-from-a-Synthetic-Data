
from config import *
import os, shutil, stat
import numpy as np
import cv2

class Data_Generator():
    """
    this class contains method for generating synthetic data with and without circles in the images

    Attributes:
    perc_mix_images: float, percentage of images to be generated with mix shapes out of all the images 
                    with circles
    perc_circle_images: float, percentage of images to be generated with circles
    mix_shape_generator: boolean, an boolean whether images of mixture of shapes is to be 
                         created or not
    """


    def __init__(self, perc_mix_images= 0, perc_circle_images=0.5, add_ellipse_images = False):
        self.perc_mix_images = perc_mix_images
        self.perc_circle_images = perc_circle_images
        self.add_ellipse_images = add_ellipse_images

        # If the Images folder already exists, then delete the folder
        if 'Images' in os.listdir():
            shutil.rmtree(f"{os.getcwd()}\\Images", onerror=self.remove_read_only)


    def remove_read_only(self, func, path, excinfo):
        """
        method to remove read only files

        :param [func, path, excinfo]: various parameters for changing the mode
        """
        os.chmod(path, stat.S_IWRITE)
        func(path)


    def save_image(self, image, file_name):
        """
        method to save the image
        :param image: np.ndarray, a numpy array of image
        :param file_name: str, a string of the file name in which image is to be stored
        """

        # Create Image folder if it does not exist
        if not os.path.exists('Images'):
            os.makedirs('Images')
        cv2.imwrite(os.path.join(os.getcwd(), 'Images', file_name), image)  


    def normalize_image(self, images):
        """
        method to normalize the images
        :param images: np.ndarray, a numpy array of all the generated images
        :return images_norm: np.ndarray, a numpy array of all the normalized images
        """

        return images / 255

    def circle_parameter_generator(self):
        """
        method to generate parameters for adding circle in the image

        :return [center_coordinates, radius, color, thickness]: list, a list of all the 
                circle parameters
        """

        # Center coordinates generated randomly
        center_coordinates = (np.random.randint(15, 85), np.random.randint(15, 85))
        
        # Radius of circle generated randomly
        radius = np.random.randint(5, 25)
        
        # BGR setting generated randomly
        color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

        # Line thickness generated randomly
        thickness = np.random.randint(5)

        return center_coordinates, radius, color, thickness


    def rectangle_parameter_generator(self):
        """
        method to generate parameters for adding rectangle in the image

        :return [start_point, end_point, color, thickness]: list, a list of all the rectangle 
                parameters
        """

        # Starting co-ordinate generated randomly
        start_point = (np.random.randint(3, 25), np.random.randint(3, 25))
        
        # Ending co-ordinate generated randomly
        end_point = (np.random.randint(30, 85), np.random.randint(30, 85))
        
        # BGR setting generated randomly
        color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
        
        # Line thickness generated randomly
        thickness = np.random.randint(5)

        return start_point, end_point, color, thickness

    
    def ellipse_parameter_generator(self):
        """
        method to generate parameters for adding ellipse in the image

        :return [center_coordinates, axesLength, angles, color, thickness]: list, a list of all 
                the ellipse parameters
        """

        # Center coordinates generated randomly
        center_coordinates = (np.random.randint(15, 85), np.random.randint(15, 85))
        
        axesLength = (np.random.randint(3, 25), np.random.randint(3, 25))
        
        #  Defining angles as [angle, start angle, end angle]
        angles = [0, 0, 360]
        
        # BGR setting generated randomly
        color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
        
        # Line thickness generated randomly
        thickness = np.random.randint(5)

        return center_coordinates, axesLength, angles, color, thickness


    def triangle_parameter_generator(self):
        """
        method to generate parameters for adding triangle in the image

        :return [[p1, p2, p3], color, thickness]: list, a list of all 
                the ellipse parameters
        """
        
        # Three vertices(tuples) of the triangle 
        p1 = (np.random.randint(5, 15), np.random.randint(5, 15))
        p2 = (np.random.randint(65, 75), np.random.randint(25, 35))
        p3 = (np.random.randint(45, 75), np.random.randint(65, 85))
        
        # BGR setting generated randomly
        color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
        
        # Line thickness generated randomly
        thickness = np.random.randint(1, 4)

        return [p1, p2, p3], color, thickness


    def image_circle_generator(self):
        """
        method to generate an image with or without circles

        :return image: np.ndarray, a numpy array of image
        :return label: int, a label for indicating whether circle is present in the 
                            image or not
        """

        # Create an Image of size (100, 100, 3)
        image = np.zeros((img_config['img_height'], img_config['img_height'], 
                            img_config['no_channels']), np.uint8)

        # Get all the parameters
        center_coordinates, radius, color, thickness = self.circle_parameter_generator()

        # Draw a circle with random parameters
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
        # 1 represents circle is present in the image
        label = 1

        return image, label

    def image_rectangle_generator(self):
        """
        method to generate an image with or without circles

        :return image: np.ndarray, a numpy array of image
        :return label: int, a label for indicating whether circle is present in the 
                            image or not
        """

        image = np.zeros((img_config['img_height'], img_config['img_height'], 3), np.uint8)

        # Get all the parameters
        start_point, end_point, color, thickness = self.rectangle_parameter_generator()
        
        # Draw a rectangle with random parameters
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        # 0 represents circle is absent in the image
        label = 0

        return image, label


    def image_ellipse_generator(self):
        """
        method to generate an image with ellipse

        :return image: np.ndarray, a numpy array of image
        :return label: int, a label for indicating whether circle is present in the 
                            image or not
        """

        # Create an Image of size (100, 100, 3)
        image = np.zeros((img_config['img_height'], img_config['img_height'], 
                            img_config['no_channels']), np.uint8)

        # Get all the parameters
        center_coordinates, axesLength, angles, color, thickness = self.ellipse_parameter_generator()

        # Draw a circle with random parameters
        image = cv2.ellipse(image, center_coordinates, axesLength,
           angles[0], angles[1], angles[2], color, thickness)
        # 1 represents circle is present in the image
        label = 0

        return image, label


    def image_triangle_generator(self):
        """
        method to generate an image with triangle

        :return image: np.ndarray, a numpy array of image
        :return label: int, a label for indicating whether circle is present in the 
                            image or not
        """

        # Create an Image of size (100, 100, 3)
        image = np.zeros((img_config['img_height'], img_config['img_height'], 
                            img_config['no_channels']), np.uint8)

        # Get all the parameters
        points, color, thickness = self.triangle_parameter_generator()

        # Draw a circle with random parameters
        image = cv2.line(image, points[0], points[1], color, thickness)
        image = cv2.line(image, points[1], points[2], color, thickness)
        image = cv2.line(image, points[0], points[2], color, thickness)
        # 1 represents circle is present in the image
        label = 0

        return image, label


    def image_mix_generator(self):
        """
        method to generate an image with a mixture of circle and rectangle

        :return image: np.ndarray, a numpy array of image
        :return label: int, a label for indicating whether circle is present in the 
                            image or not
        """

        image = np.zeros((img_config['img_height'], img_config['img_height'], 3), np.uint8)

        # Get all the parameters and draw a circle
        center_coordinates, radius, color, thickness = self.circle_parameter_generator()
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
        
        # Ellipse inserted only if add_ellipse_images = True
        if self.add_ellipse_images:
            # Here flag = 2 represents a flag of inserting Ellipse
            flag = np.random.randint(3)
        else:
            flag = np.random.randint(2)
        if flag == 0:  
            # If flag is 0, then insert Rectangle 
            start_point, end_point, color, thickness = self.rectangle_parameter_generator()
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
        elif flag == 1:
            # If flag is 1, then insert TrianglE
            # Get all the parameters and insert Triangle
            points, color, thickness = self.triangle_parameter_generator()
            # Draw a circle with random parameters
            image = cv2.line(image, points[0], points[1], color, thickness)
            image = cv2.line(image, points[1], points[2], color, thickness)
            image = cv2.line(image, points[0], points[2], color, thickness)
        else:
            # If flag is "2", then insert Ellipse
            center_coordinates, axesLength, angles, color, thickness = self.ellipse_parameter_generator()
            image = cv2.ellipse(image, center_coordinates, axesLength,
            angles[0], angles[1], angles[2], color, thickness)

        # 1 represents circle is present in the image
        label = 1

        return image, label


    def image_generator(self):
        """
        method to generate no_img images

        :return image_array: np.ndarray, a numpy array of all the generated images
        :return label_array: list, a list of all the labels
        """

        no_images_with_circle = int(img_config['no_img'] * self.perc_circle_images)
        no_mix_images = int(no_images_with_circle * self.perc_mix_images)
        
        image_array = np.zeros((img_config['no_img'], img_config['img_height'], 
                                img_config['img_width'], img_config['no_channels']))
        label_array = []
        print('Total Images to be created is:', img_config['no_img'])
        print('Images with circle is:', no_images_with_circle)
        print('Images with mix shapes and circle is:', no_mix_images)
        print('Creating Images...')
        for idx in range(img_config['no_img']):
            # Create Images according to the percentages given by the user
            if idx < no_mix_images:
                image, label = self.image_mix_generator()
            elif idx < no_images_with_circle:
                image, label = self.image_circle_generator()
            else:
                # Ellipse inserted only if add_ellipse_images = True
                if self.add_ellipse_images:
                    # Here flag = 2 represents a flag of inserting Ellipse
                    flag = np.random.randint(3)
                else:
                    flag = np.random.randint(2)
                if flag == 0:  
                    # If flag is 0, then insert Rectangle
                    image, label = self.image_rectangle_generator()
                elif flag == 1:
                    # If flag is 1, then insert Triangle
                    image, label = self.image_triangle_generator()
                else:
                     # If flag is 2, then insert Ellipse
                    image, label = self.image_ellipse_generator()
                
            file_name = f"Img_{str(idx).zfill(4)}.png"
            self.save_image(image, file_name)
            image_array[idx] = image
            label_array.append(label)
        print('Images created and saved in directory as below:-')
        print(os.getcwd())
        return self.normalize_image(image_array), label_array
    
