import cv2
import numpy as np

img_list = [('1.jpg','2.jpg'),
			('hill1.jpg','hill2.jpg'),
			('S1.jpg','S2.jpg')]

# use cv2 sift_create to find the interests points
# output image with name 'sift_image_name.jpg' just to check the result
# return kps = keypoints, des = descriptors
def sift(input_img, img_name):
	img = np.uint8(input_img)
	descriptor = cv2.xfeatures2d.SIFT_create()
	(kps, des) = descriptor.detectAndCompute(img, None)
	cv2.drawKeypoints(img, kps, img, (0,255,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite('./data/sift_{}.jpg'.format(img_name.split('.')[0]),img)
	return kps, des

def find_match(des1, des2):
	dist_matrix = np.zeros((len(des1), len(des2)))
	# find the distance between each and every descriptors
	for index1 in range(len(des1)):
		for index2 in range(len(des2)):
			dist = np.linalg.norm(des1[index1] - des2[index2])
			dist_matrix[index1][index2] = dist

	#sort the vector to get the two best match with the smallest distance
	#match = [descriptor's number(descriptor i), the two best match with descriptor i, distance with the first match, destance with the second match]
	match = []
	for i in range(dist_matrix.shape[0]):
		two_best_match = np.argsort(dist_matrix[i])[:2]
		match.append ([i, two_best_match, dist_matrix[i][two_best_match[0]], dist_matrix[i][two_best_match[1]]])

	#perform ratio distance with ratio = 0.75
	final_match = []
	for i in range(len(match)):
		if match[i][2] < match[i][3] * 0.75:
			final_match.append([match[i][0], match[i][1][0]])
	return final_match


def draw_match(img1, img2, kps1, kps2, match, img_name):
	height1 = img1.shape[0]
	width1 = img1.shape[1]
	height2 = img2.shape[0]
	width2 = img2.shape[1]

	output_img = np.zeros((max(height1,height2), width1+width2, 3), dtype = 'uint8')
	output_img[0:height1, 0:width1] = img1
	output_img[0:height2, width1:] = img2

	for index1, index2 in match:
		color = list(map(int, np.random.randint(0, high=255, size=(3,))))
		pts1 = (int(kps1[index1].pt[0]), int(kps1[index1].pt[1]))
		pts2 = (int(kps2[index2].pt[0] + width1), int(kps2[index2].pt[1]))
		cv2.line(output_img, pts1, pts2, color, 1)

	cv2.imwrite('./data/matchline_'+img_name+'.jpg', output_img)


for img in img_list:
	img1 = cv2.imread('./data/{}'.format(img[0]))
	img2 = cv2.imread('./data/{}'.format(img[1]))
	
	# kps = keypoints, des = descriptor
	kps1, des1 = sift(cv2.imread('./data/{}'.format(img[0])), img[0])
	kps2, des2 = sift(cv2.imread('./data/{}'.format(img[1])), img[1])

	# feature matching
	match = find_match(des1, des2)

	# draw the matching result (by drawing lines between mathing points) to check the result
	img_name = img[0].split('.')[0] + '_' + img[1].split('.')[0]
	draw_match(img1, img2, kps1, kps2, match, img_name)

# find homography matrix H
def homomat(min_match_count: int, src, dst):
    A = np.zeros((min_match_count * 2, 9))
    # construct the two sets of points
    for i in range(min_match_count):
        src1, src2 = src[i, 0, 0], src[i, 0, 1]
        dst1, dst2 = dst[i, 0, 0], dst[i, 0, 1]
        A[i * 2, :] = [src1, src2, 1, 0, 0, 0, -src1 * dst1, - src2 * dst1, -dst1]
        A[i * 2 + 1, :] = [0, 0, 0, src1, src2, 1, -src1 * dst2, - src2 * dst2, -dst2]
    
	# compute the homography between the two sets of points
    [_, S, V] = np.linalg.svd(A)
    m = V[np.argmin(S)]
    m *= 1 / m[-1]
    H = m.reshape((3, 3))
    return H


def ransac(final_match, kps_list, min_match_count, num_test: int, threshold: float):
    if len(final_match) > min_match_count:
        src_pts = np.array([kps_list[1][m[1]].pt for m in final_match]).reshape(-1, 1, 2)
        dst_pts = np.array([kps_list[0][m[0]].pt for m in final_match]).reshape(-1, 1, 2)
        min_outliers_count = math.inf
        
        while(num_test != 0):
            indexs = np.random.choice(len(final_match), min_match_count, replace=False)
            homography = homomat(min_match_count, src_pts[indexs], dst_pts[indexs])

            # Warp all left points with computed homography matrix and compare SSDs
            src_pts_reshape = src_pts.reshape(-1, 2)
            one = np.ones((len(src_pts_reshape), 1))
            src_pts_reshape = np.concatenate((src_pts_reshape, one), axis=1)
            warped_left = np.array(np.mat(homography) * np.mat(src_pts_reshape).T)
            for i, value in enumerate(warped_left.T):
                warped_left[:, i] = (value * (1 / value[2])).T

            # Calculate SSD
            dst_pts_reshape = dst_pts.reshape(-1, 2)
            dst_pts_reshape = np.concatenate((dst_pts_reshape, one), axis=1)
            inlier_count = 0
            inlier_list = []
            for i, pair in enumerate(final_match):
                ssd = np.linalg.norm(np.array(warped_left[:, i]).ravel() - dst_pts_reshape[i])
                if ssd <= threshold:
                    inlier_count += 1
                    inlier_list.append(pair)

            if (len(final_match) - inlier_count) < min_outliers_count:
                min_outliers_count = (len(final_match) - inlier_count)
                best_homomat = homography
                best_matches = inlier_list
            num_test -= 1
        return best_homomat, best_matches
    else:
        raise Exception("Not much matching keypoints exits!")



