import pdb
import glob
import cv2
import os
import numpy as np

from skimage import exposure

class PanaromaStitcher():
    def __init__(self):
        pass


    def compute_homography(self, img1_pts, img2_pts):
        """
        Computes the homography matrix between two images.

        Parameters
        ----------
        img1_pts : numpy.ndarray
            Points from the first image.

        img2_pts : numpy.ndarray
            Points from the second image.

        Returns
        -------

        numpy.ndarray
            Homography matrix.
        """
        A = []
        for i in range(len(img1_pts)):
            x1, y1,z1 = img1_pts[i]
            x2, y2,z2 = img2_pts[i]
            A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
            A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])

        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape(3,3)
        H = H/H[-1,-1]
        return H
    
    def ransac(self, img1_pts, img2_pts, threshold=10.0, max_iter=1500):
        """
        RANSAC algorithm to estimate the homography matrix between two images.

        Parameters
        ----------
        img1_pts : numpy.ndarray
            Points from the first image.

        img2_pts : numpy.ndarray
            Points from the second image.

        threshold : float
            Maximum distance threshold for inliers.
            default: 10.0

        max_iter : int
            Maximum number of iterations.
            default: 1500

        Returns
        -------

        numpy.ndarray
            Homography matrix.
        """

        max_inliers = 0
        best_H = None
        best_inliers = []

        for _ in range(max_iter):
            idx = np.random.choice(len(img1_pts), 4, replace=False)
            H = self.compute_homography(img1_pts[idx], img2_pts[idx])

            # Skip if homography is degenerate
            if not np.isfinite(H).all():
                continue

            inliers = []
            for j in range(len(img1_pts)):
                pt1 = np.array([img1_pts[j][0], img1_pts[j][1], 1])
                pt2 = np.array([img2_pts[j][0], img2_pts[j][1], 1])
                pt2_ = np.dot(H, pt1)

                pt2_ = pt2_ / pt2_[-1]  # Normalize homogeneous coordinates

                if np.linalg.norm(pt2 - pt2_) < threshold:
                    inliers.append(j)

            # Update if current inliers exceed the max found so far
            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_inliers = inliers  
                best_H = H

        # Optionally refine the homography using all inliers . 
        # Did not use this as it was not giving good results
        # if best_inliers:
        #     best_H = self.compute_homography(img1_pts[best_inliers], img2_pts[best_inliers])

        return best_H

    def calculate_warped_shape(self, img, H):
        """
        Calculates the shape of the warped image after applying a homography matrix.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be warped.
        H : numpy.ndarray
            Homography matrix.

        Returns
        -------
        translation_matrix : numpy.ndarray
            Translation matrix.

        updated_H : numpy.ndarray   
            Updated homography matrix after applying the translation matrix to the original homography matrix.

        (min_x, max_x, min_y, max_y) : tuple
            Coordinates of the warped image.
        """
        h, w = img.shape[:2]
        corners = np.array([[0, 0, 1], [0, h, 1], [w, 0, 1], [w, h, 1]])
        corners = np.dot(H, corners.T).T
        corners = corners / corners[:, 2][:, None]
        corners = corners[:, :2]

        min_x = int(np.floor(min(corners[:, 0])))
        max_x = int(np.ceil(max(corners[:, 0])))
        min_y = int(np.floor(min(corners[:, 1])))
        max_y = int(np.ceil(max(corners[:, 1])))

        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        updated_H = np.dot(translation_matrix, H)

        return translation_matrix,updated_H,(min_x, max_x, min_y, max_y)
    

    def cylindricalWarp(self, img, f):
        """
        Applies a cylindrical warp to the input image using a specified focal length.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be warped.
        f : float
            Focal length to use for warping.

        Returns
        -------
        numpy.ndarray
            The warped image.

        Notes
        -----
        This function has been adapted with minor modifications from:\n
        https://www.morethantechnical.com/blog/2018/10/30/cylindrical-image-warping-for-panorama-stitching/
        """
        h_,w_ = img.shape[:2]
        K = np.array([[f,0,w_/2],[0,f,h_/2],[0,0,1]]) # mock intrinsics
        # pixel coordinates
        y_i, x_i = np.indices((h_,w_))
        X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
        Kinv = np.linalg.inv(K) 
        X = Kinv.dot(X.T).T # normalized coords
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
        B = K.dot(A.T).T # project back to image-pixels plane
        # back from homog coords
        B = B[:,:-1] / B[:,[-1]]
        # make sure warp coords only within image bounds
        B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
        B = B.reshape(h_,w_,-1)

        img_bgr = img  # Use the original BGR image
        return cv2.remap(img_bgr, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    

    def stitch(self, img1, img2, H):
        """ 
        This function stiches two images using the homography matrix H.

        Parameters
        ----------
        img1: numpy array
            The first image
        img2: numpy array
            The second image
        H: numpy array
            The homography matrix

        Returns
        -------
        canvas: numpy array
            The stiched image
        """

        # Calculate the warped shape of img1 using the homography matrix H and the translation matrix
        translational_matrix, updated_H, (min_x, max_x, min_y, max_y) = self.calculate_warped_shape(img1, H)

        # Set up canvas
        canvas_x_min = min(min_x, 0)                        # Minimum x coordinate of the canvas
        canvas_x_max = max(max_x, img2.shape[1])            # Maximum x coordinate of the canvas
        canvas_y_min = min(min_y, 0)                        # Minimum y coordinate of the canvas
        canvas_y_max = max(max_y, img2.shape[0])            # Maximum y coordinate of the canvas

        # Create an empty canvas to place the images
        canvas = np.zeros((canvas_y_max - canvas_y_min, canvas_x_max - canvas_x_min, 3), dtype=np.uint8)
        
        # Place img2 on the canvas
        x_offset, y_offset = -canvas_x_min, -canvas_y_min
        canvas[y_offset:y_offset + img2.shape[0], x_offset:x_offset + img2.shape[1]] = img2 # Place img2 on the canvas after applying the translation

        # Generate all coordinates in the canvas
        canvas_h, canvas_w = canvas.shape[:2]
        y_coords, x_coords = np.indices((canvas_h, canvas_w))

        # Stack coordinates and transform using the inverse of updated_H
        homogenous_coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords).ravel()])  # Homogenous coordinates
        transformed_coords = np.linalg.inv(updated_H) @ homogenous_coords                                   # Transform using the inverse of updated_H
        transformed_coords /= transformed_coords[2, :]                                                      # Normalize

        # Map coordinates back to img1
        x_transformed = transformed_coords[0, :].round().astype(int)                                        # Round and convert to int
        y_transformed = transformed_coords[1, :].round().astype(int)                                        # Round and convert to int

        # Create a mask for valid img1 coordinates
        valid_mask = (0 <= x_transformed) & (x_transformed < img1.shape[1]) & (0 <= y_transformed) & (y_transformed < img1.shape[0])

        # Apply mask to get valid canvas and img1 coordinates
        canvas_x_coords = x_coords.ravel()[valid_mask]
        canvas_y_coords = y_coords.ravel()[valid_mask]
        img1_x_coords = x_transformed[valid_mask]
        img1_y_coords = y_transformed[valid_mask]

        # Assign pixels from img1 to canvas where the mask is valid, using vectorized operations
        canvas_pixels = canvas[canvas_y_coords, canvas_x_coords]
        img1_pixels = img1[img1_y_coords, img1_x_coords]

        # Blend only where necessary
        canvas[canvas_y_coords, canvas_x_coords] = np.where(
            (canvas_pixels == 0).all(axis=1, keepdims=True),
            img1_pixels,
            canvas_pixels
        )

        return canvas



    def stich_two_images(self, img1, img2):
        """
        This function stiches two images together.

        Parameters
        ----------
        img1: numpy array
            The first image
        img2: numpy array
            The second image

        Returns
        -------
        H: numpy array
            The homography matrix

        stitched_image: numpy array 
            The stiched image
        """

        # Convert images to grayscale to extract features and match them
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1_gray,None)                   # Get the keypoints and descriptors for image 1
        kp2, des2 = sift.detectAndCompute(img2_gray,None)                   # Get the keypoints and descriptors for image 2

        # Using FLANN based matcher to match the features
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)          # Setting the index parameters
        search_params = dict(checks=50)                                     # Setting the search parameters
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test to get good matches
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:                                # If the distance of the first match is less than
                good.append(m)                                              # 0.75 times the distance of the second match, then it is a good match

        # Get the points for the good matches
        img1_pts = []
        img2_pts = []
        for m in good:
            img1_pts.append(kp1[m.queryIdx].pt)
            img2_pts.append(kp2[m.trainIdx].pt)

        # Convert the points to float32
        img1_pts = np.float32(img1_pts).reshape(-1,2)
        img1_pts = np.concatenate([img1_pts, np.ones((img1_pts.shape[0],1))], axis=1)
        img2_pts = np.float32(img2_pts).reshape(-1,2)
        img2_pts = np.concatenate([img2_pts, np.ones((img2_pts.shape[0],1))], axis=1)

        # Get the homography matrix using RANSAC
        H = self.ransac(img1_pts, img2_pts)

        # Stitch the images using the homography matrix
        stitched_image = self.stitch(img1, img2, H)

        # Return the homography matrix and the stiched image
        return H,stitched_image

    def make_panaroma_for_images_in(self,path,f):
        
        """
        This function creates a panaroma for the images in the specified path. \n
        Final panaroma is obtained by stiching the images from the middle to the left and right.

        Parameters
        ----------
        path: str
            Path to the images
        f: float
            Focal length for the cylindrical warp

        Returns
        -------
        all_stiched_images: list
            List of all stiched images

        homography_matrix_list: list
            List of all homography matrices

        Note
        -----
        This function was given in the assignment prompt.
        """

        # Set the path to the images for panaroma stitching
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        n = len(all_images)

        # Read all images
        for idx, img in enumerate(all_images):
            all_images[idx] = cv2.imread(img)
            print('\t\t reading... {} having size {}'.format(img,all_images[idx].shape))

        # Apply cylindrical warp
        for idx,img in enumerate(all_images):
            all_images[idx] = self.cylindricalWarp(all_images[idx],f)
        
        # We will store al intermediate stiched images as well as the homography matrices
        all_stiched_images = []                                             # Stores all intermediate stiched images
        homography_matrix_list =[]                                          # Stores all homography matrices
        idx = 1                                                             # Index to keep track of the intermediate stiched images

        # We will start stiching the images from the middle
        # If the number of images is even, we will stich the middle two images first
        # If the number of images is odd, we will consider the middle image as the first stiched image
        # After recieving the first stiched image, we will stich the images from the middle to the left and right
        if n%2 == 0:
            h,stitched_image = self.stich_two_images(all_images[n//2-1], all_images[n//2])
            left = all_images[:len(all_images)//2-1]                        # Left images from the middle
            right = all_images[len(all_images)//2+1:]                       # Right images from the middle
            right = right[::-1]                                             # Reverse the right images to maintain the order
            homography_matrix_list.append(h)                                # Append the homography matrix
            print('stiched_image {} pair of images'.format(idx))
            idx += 1
        else:
            stitched_image = all_images[len(all_images)//2]                 # Middle image is the first stiched image
            left = all_images[:len(all_images)//2]                          # Left images from the middle
            right = all_images[len(all_images)//2+1:]                       # Right images from the middle
            right = right[::-1]                                             # Reverse the right images to maintain the order

        all_stiched_images.append(stitched_image)                           # Append the first stiched image

        while len(left) > 0 and len(right) > 0:                             # Loop to stich the images from the middle to the left and right

            h1,stitched = self.stich_two_images(left.pop(),stitched_image)  # Stich the left image with the current stiched image
            all_stiched_images.append(stitched)                             # Append the stiched image
            print('stiched_image {} pair of images'.format(idx))            
            idx += 1
            h2,stitched = self.stich_two_images(stitched, right.pop())      # Stich the right image with the current stiched image
            all_stiched_images.append(stitched)                             # Append the stiched image
            print('stiched_image {} pair of images'.format(idx))
            idx += 1
            homography_matrix_list.append(h1)                               # Append the homography matrix for the left image
            homography_matrix_list.append(h2)                               # Append the homography matrix for the right image
            stitched_image = stitched.copy()                                # Update the current stiched image

        # Stich the remaining images if any from the left
        # I believe this will never be the case as we are considering the middle image as the first stiched image
        # But still added this for the sake of completeness
        if len(left) > 0:           
            h,stitched = self.stich_two_images(left.pop(),stitched_image)   # Stich the left image with the current stiched image
            print('stiched_image {} pair of images'.format(idx))
            idx += 1
            homography_matrix_list.append(h)                                # Append the homography matrix
            stitched_image = stitched.copy()                                # Update the current stiched image
            all_stiched_images.append(stitched)                             # Append the stiched image
        
        # Stich the remaining images if any from the right
        # I believe this will never be the case as we are considering the middle image as the first stiched image
        # But still added this for the sake of completeness
        if len(right) > 0:
            h,stitched = self.stich_two_images(stitched_image, right.pop()) # Stich the right image with the current stiched image
            print('stiched_image {} pair of images'.format(idx))
            idx += 1
            homography_matrix_list.append(h)                                # Append the homography matrix
            stitched_image = stitched.copy()                                # Update the current stiched image
            all_stiched_images.append(stitched)                             # Append the stiched image

        return all_stiched_images, homography_matrix_list                   # Return the stiched images and homography matrices

if __name__ == '__main__':

    np.random.seed(3471)

    # focal_pixels = (focal_mm * image_width_pixels) / sensor_width_mm
    # focal mm and image width are obtained from the image metadata
    # sensor width is obtained from the camera specs by googling the camera model

    focal_pixels = {"I1": (14.0 * 1632) / 5.74,
                    "I2": (5.55 * 653) / 5.74,
                    "I3": (5.725 * 730)/ 6.17,
                    "I4": (25.5 * 2000) / 23.55,
                    "I5": (24.0 * 2000) / 23.55,
                    "I6": (30.0 * 602)/ 23.7}

    # Set the path to the images for testing purposes
    # Below code is for testing the stitcher and has been adopted from the main.py
    path = 'test_imgs{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters

    inst = PanaromaStitcher()
    for impaths in glob.glob(path):
        print('\t\t Processing... {}'.format(impaths))

        # Get the focal length
        folder = impaths.split(os.sep)[-1]
        f=focal_pixels[folder]

        # Get the stitched images and homography matrices
        stitched_images, homography_matrix_list = inst.make_panaroma_for_images_in(path=impaths,f=f)

        # Stores the Final Output
        outfile =  './results/{}/{}.png'.format(impaths.split(os.sep)[-1],inst.__class__.__name__)
        os.makedirs(os.path.dirname(outfile),exist_ok=True)

        # Stores the intermediate stichted images
        intermediate_dir = './intermediate/{}'.format(impaths.split(os.sep)[-1])
        os.makedirs(intermediate_dir,exist_ok=True)

        # Loop to store intermediate images
        for i in range(len(stitched_images)):
            cv2.imwrite('./results/{}/{}.png'.format(impaths.split(os.sep)[-1],"stitched_image_"+str(i)),stitched_images[i])

        # Save the final panaroma
        cv2.imwrite(outfile,stitched_images[-1])

        # Print the homography matrices
        print(homography_matrix_list)
        print(len(homography_matrix_list))
        print('Panaroma saved ... @ ./results/{}.png'.format(inst.__class__.__name__))
        print('\n\n')
