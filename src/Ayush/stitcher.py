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

        # Optionally refine the homography using all inliers
        # if best_inliers:
        #     best_H = self.compute_homography(img1_pts[best_inliers], img2_pts[best_inliers])

        return best_H

    def calculate_warped_shape(self, img, H):
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
    
    def cylindricalWarp(self,img, f):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
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
        
        # img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
        # warp the image according to cylindrical coords
        # return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

        img_bgr = img  # Use the original BGR image
        return cv2.remap(img_bgr, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    

    def stitch(self, img1, img2, H):
        # Step 1: Calculate canvas boundaries and updated homography
        translational_matrix, updated_H, (min_x, max_x, min_y, max_y) = self.calculate_warped_shape(img1, H)

        # Step 2: Set up canvas
        canvas_x_min = min(min_x, 0)
        canvas_x_max = max(max_x, img2.shape[1])
        canvas_y_min = min(min_y, 0)
        canvas_y_max = max(max_y, img2.shape[0])

        # Create canvas
        canvas = np.zeros((canvas_y_max - canvas_y_min, canvas_x_max - canvas_x_min, 3), dtype=np.uint8)
        
        # Place img2 on the canvas
        x_offset, y_offset = -canvas_x_min, -canvas_y_min
        canvas[y_offset:y_offset + img2.shape[0], x_offset:x_offset + img2.shape[1]] = img2

        # Step 3: Generate all coordinates in the canvas
        canvas_h, canvas_w = canvas.shape[:2]
        y_coords, x_coords = np.indices((canvas_h, canvas_w))

        # Stack coordinates and transform using the inverse of updated_H
        homogenous_coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords).ravel()])
        transformed_coords = np.linalg.inv(updated_H) @ homogenous_coords
        transformed_coords /= transformed_coords[2, :]  # Normalize

        # Map coordinates back to img1
        x_transformed = transformed_coords[0, :].round().astype(int)
        y_transformed = transformed_coords[1, :].round().astype(int)

        # Step 4: Create a mask for valid img1 coordinates
        valid_mask = (0 <= x_transformed) & (x_transformed < img1.shape[1]) & (0 <= y_transformed) & (y_transformed < img1.shape[0])

        # Apply mask to get valid canvas and img1 coordinates
        canvas_x_coords = x_coords.ravel()[valid_mask]
        canvas_y_coords = y_coords.ravel()[valid_mask]
        img1_x_coords = x_transformed[valid_mask]
        img1_y_coords = y_transformed[valid_mask]

        # Step 5: Assign pixels from img1 to canvas where the mask is valid, using vectorized operations
        canvas_pixels = canvas[canvas_y_coords, canvas_x_coords]
        img1_pixels = img1[img1_y_coords, img1_x_coords]

        # Blend only where necessary
        canvas[canvas_y_coords, canvas_x_coords] = np.where(
            (canvas_pixels == 0).all(axis=1, keepdims=True),
            img1_pixels,
            canvas_pixels
        )

        return canvas


    # def stitch(self, img1, img2, H):

    #     translational_matrix,updated_H, (min_x, max_x, min_y, max_y) = self.calculate_warped_shape(img1, H)

    #     canvas_x_min = min(min_x, 0)    
    #     canvas_x_max = max(max_x, img2.shape[1])
    #     canvas_y_min = min(min_y, 0)
    #     canvas_y_max = max(max_y, img2.shape[0])

    #     canvas = np.zeros((canvas_y_max - canvas_y_min, canvas_x_max - canvas_x_min, 3), dtype=np.uint8)

    #     canvas[-canvas_y_min:-canvas_y_min + img2.shape[0], -canvas_x_min:-canvas_x_min + img2.shape[1]] = img2

    #     for i in range(canvas.shape[0]):
    #         for j in range(canvas.shape[1]):
    #             pt1 = np.dot(np.linalg.inv(updated_H), np.array([j, i, 1]))
    #             pt1 = pt1 / pt1[-1]
    #             x, y = int(pt1[0]), int(pt1[1])

    #             if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:

    #                 if np.all(canvas[i, j] == 0):
    #                     canvas[i, j] = img1[y, x]
    #                 else:
    #                     # print(canvas[i, j].shape, img1[y, x].shape)
    #                     canvas[i, j] = img1[y, x]//2 + canvas[i, j]//2
                        
    #     canvas = canvas.astype(np.uint8)
    #     return canvas

    def stich_two_images(self, img1, img2):

        # img1 = exposure.match_histograms(img1, ref_img,channel_axis=-1)
        # img2 = exposure.match_histograms(img2, ref_img,channel_axis=-1)

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1_gray,None)
        kp2, des2 = sift.detectAndCompute(img2_gray,None)

        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des1,des2, k=2)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        img1_pts = []
        img2_pts = []
        for m in good:
            img1_pts.append(kp1[m.queryIdx].pt)
            img2_pts.append(kp2[m.trainIdx].pt)

        img1_pts = np.float32(img1_pts).reshape(-1,2)
        img1_pts = np.concatenate([img1_pts, np.ones((img1_pts.shape[0],1))], axis=1)
        img2_pts = np.float32(img2_pts).reshape(-1,2)
        img2_pts = np.concatenate([img2_pts, np.ones((img2_pts.shape[0],1))], axis=1)

        H = self.ransac(img1_pts, img2_pts)

        pts1_ = np.dot(H, img1_pts.T).T
        pts1_ = pts1_ / pts1_[-1,-1]
    
        stitched_image = self.stitch(img1, img2, H)

        return H,stitched_image

    def make_panaroma_for_images_in(self,path,f):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        print("focal lenght for images is {}".format(f))
        n = len(all_images)

        for idx, img in enumerate(all_images):
            all_images[idx] = cv2.imread(img)
            print('\t\t reading... {} having size {}'.format(img,all_images[idx].shape))

        # if n%2 == 0:
        #     ref_img = all_images[n//2-1].copy()
        # else:
        #     ref_img = all_images[len(all_images)//2].copy()

        # ref_img= all_images[-1].copy()

        for idx,img in enumerate(all_images):
            # all_images[idx] = exposure.match_histograms(all_images[idx], ref_img,channel_axis=-1)
            all_images[idx] = self.cylindricalWarp(all_images[idx],f)
            

        all_stiched_images = []
        homography_matrix_list =[]
        idx = 1
        if n%2 == 0:
            print('stiched_image {} pair of images'.format(idx))
            # ref_img = all_images[n//2-1].copy()
            h,stitched_image = self.stich_two_images(all_images[n//2-1], all_images[n//2])
            left = all_images[:len(all_images)//2-1]
            right = all_images[len(all_images)//2+1:]
            right = right[::-1]
            homography_matrix_list.append(h)
            idx += 1
        else:
            ref_img = all_images[len(all_images)//2].copy()
            stitched_image = all_images[len(all_images)//2]
            left = all_images[:len(all_images)//2]
            right = all_images[len(all_images)//2+1:]
            right = right[::-1]

        all_stiched_images.append(stitched_image)

        while len(left) > 0 and len(right) > 0:

            h1,stitched = self.stich_two_images(left.pop(),stitched_image)
            all_stiched_images.append(stitched)
            print('stiched_image {} pair of images'.format(idx))
            idx += 1
            h2,stitched = self.stich_two_images(stitched, right.pop())
            all_stiched_images.append(stitched)
            print('stiched_image {} pair of images'.format(idx))
            idx += 1
            homography_matrix_list.append(h1)
            homography_matrix_list.append(h2)
            stitched_image = stitched.copy()

        if len(left) > 0:
            h,stitched = self.stich_two_images(left.pop(),stitched_image)
            print('stiched_image {} pair of images'.format(idx))
            idx += 1
            homography_matrix_list.append(h)
            stitched_image = stitched.copy()
            all_stiched_images.append(stitched)

        if len(right) > 0:
            h,stitched = self.stich_two_images(stitched_image, right.pop())
            print('stiched_image {} pair of images'.format(idx))
            idx += 1
            homography_matrix_list.append(h)
            stitched_image = stitched.copy()
            all_stiched_images.append(stitched)

        return all_stiched_images, homography_matrix_list 

if __name__ == '__main__':

    np.random.seed(3471)
    # np.random.seed(3407)

    # These informations have been obtained from the camera models and the meta info from the images
    # I then tweaked them a little to get the best results
    focal_pixels = {"I1": (16.0 * 3264) / 5.74,
                    "I2": (5.55 * 653) / 5.74,
                    "I3": (5.725 * 730)/ 6.17,
                    "I4": (25.5 * 2000) / 23.55,
                    "I5": (24.0 * 2000) / 23.55,
                    "I6": (30.0 * 602)/ 23.7}

    # Test your implementation here
    # You can change the path to test your implementation on different images
    # path = 'test_imgs{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters
    path = 'test_imgs{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters

    inst = PanaromaStitcher()
    for impaths in glob.glob(path):
        print('\t\t Processing... {}'.format(impaths))
        folder = impaths.split(os.sep)[-1]
        f=focal_pixels[folder]
        stitched_images, homography_matrix_list = inst.make_panaroma_for_images_in(path=impaths,f=f)

        outfile =  './results/{}/{}.png'.format(impaths.split(os.sep)[-1],inst.__class__.__name__)
        os.makedirs(os.path.dirname(outfile),exist_ok=True)
        cv2.imwrite(outfile,stitched_images[-1])
        for i in range(len(stitched_images)):
            cv2.imwrite('./results/{}/{}.png'.format(impaths.split(os.sep)[-1],"stitched_image_"+str(i)),stitched_images[i])
        print(homography_matrix_list)
        print(len(homography_matrix_list))
        print('Panaroma saved ... @ ./results/{}.png'.format(inst.__class__.__name__))
        print('\n\n')