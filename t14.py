import cv2
import numpy as np

from math import sin, cos

window = (720, 360)

# source points - in 3d
src_rect = np.array([[-2, 1, 0], [2, 1, 0], [2, -1, 0], [-2, -1, 0]])
# destination points - projected to 2d (this updates every frame btw) -- for aske of this sample, it will be a normal rect
# dst_rect = np.array([[window[0] // 4, window[1] // 4] , [window[0] * 3 // 4, window[1] // 4], [window[0] * 3 // 4, window[1] * 3 // 4], [window[0] // 4, window[1] * 3 // 4]])
pi4 = np.pi / 4
pi2 = np.pi/2
hx, hy = window[0]//2, window[1]//2
dst_rect = np.array([
    [hx + 50 + cos(0) * 100, hy + 100 - sin(0) * 100], 
    [hx + 100 + cos(pi2) * 100, hy + 100 - sin(pi2) * 100],
    [hx + 100 + cos(pi2 * 2) * 100, hy + 100 - sin(pi2 * 2) * 100],
    [hx + 50 + cos(pi2 * 3) * 100, hy + 100 - sin(pi2 * 3) * 100]
])

# calcualte homography matrix - opencv
hmat, status = cv2.findHomography(src_rect, dst_rect)

# create a opencv surface 
cv2_surface = np.zeros((window[1], window[0], 3), dtype=np.uint8)
# draw the dst_rect onto surface
cv2.polylines(cv2_surface, [dst_rect.astype(np.int32)], True, (0, 255, 0), 2)


# draw a point on the surface
cv2.circle(cv2_surface, (100, 100), 5, (255, 255, 255), -1)


xvec, yvec, zvec, wvec = 0.11,-0.01,-0.14,0.08

# matrices
mat1 = np.array([
    [wvec, zvec, -yvec, xvec],
    [-zvec, wvec, xvec, yvec],
    [yvec, -xvec, wvec, zvec],
    [-xvec, -yvec, -zvec, wvec]
])
mat2 = np.array([
    [wvec, zvec, -yvec, -xvec],
    [-zvec, wvec, xvec, -yvec],
    [yvec, -xvec, wvec, -zvec],
    [xvec, yvec, zvec, wvec]
])

print(mat1, mat2)
rmat = np.matmul(mat1, mat2)
print(rmat)



# show image
cv2.imshow("cv2_surface", cv2_surface)
cv2.waitKey(0)

print(hmat)


