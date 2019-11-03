import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

debug = False

def project_points(points,P):
    points = P @ points.T
    return points

def getParallaxMap(disparity):
    if(debug):
        print(disparity.shape)
    h,w = disparity.shape
    disparity_map = []
    for i in range(h):
        for j in range(w):
            disparity_map.append([j,i,disparity[i,j],1])
    return np.array(disparity_map)

def cal_disparity(image_left,image_right):
    window_size = 5
    min_disp = -39
    num_disp = 144
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        disp12MaxDiff = 1,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        preFilterCap=63
        )
    disparity = stereo.compute(image_left, image_right).astype(np.float32) / 64.0
    disparity = (disparity-min_disp)/num_disp
    return disparity

def reprojectImageTo3d(disparity_map,Q):
    points = []
    for dis in disparity_map:
        point = Q.dot(dis)
        points.append(point)
    return np.array(points)

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

folder_for_left_images = './mr19-assignment3-data/img2/00000004'
folder_for_right_images = './mr19-assignment3-data/img3/00000004'
image_extension = '.png'
start_number = 60
number_of_images = 21

ground_truth_file = './mr19-assignment3-data/poses.txt'
file_ground_truth = open(ground_truth_file,'r')

projection_matrix = []
row_init = np.array([0,0,0,1])
for line in file_ground_truth:
    p = []
    x = line.split(' ')
    for i in x:
        p.append(float(i))
    p = np.array(p).reshape((3,4))
    p = np.vstack((p,row_init))
    projection_matrix.append(p)

f = 7.070912e+02;
b = 0.53790448812;
K = np.array([[7.070912e+02, 0.000000e+00, 6.018873e+02],
              [0.000000e+00, 7.070912e+02, 1.831104e+02],
              [0.000000e+00, 0.000000e+00, 1.000000e+00]])

final_points = []
final_colors = []
pcds = []
for image_number in range(0,number_of_images,1):

    path_for_left_image = (folder_for_left_images +
                           str(start_number + image_number) +
                           image_extension)

    path_for_right_image = (folder_for_right_images +
                            str(start_number + image_number) +
                            image_extension)


    if(debug):
        print (path_for_left_image)
        print("Loading Images")

    left_image_with_color = cv2.imread(path_for_left_image)
    left_image = cv2.imread(path_for_left_image)
    right_image = cv2.imread(path_for_right_image)

    if(debug):
        print("Computing Disparity")

    disparity = cal_disparity(left_image,right_image)
    h,w = disparity.shape
    Q = np.float32([[ 1,  0,  0, -w/2],
                    [ 0,  -1,  0, h/2],
                    [ 0,  0,  0, f],
                    [ 0,  0,  1/b, 0]])

    if(debug):
        print("Computing PointCloud")

    disparity_map = getParallaxMap(disparity)
    point_cloud = reprojectImageTo3d(disparity_map,Q)
    colors = cv2.cvtColor(left_image_with_color, cv2.COLOR_BGR2RGB)
    mask = disparity >= disparity.min()
    colors = colors[mask]
    colors = colors / 255

    if(debug):
        print("Reprojecting 3D points")

    N,three = (point_cloud.shape)
    for i in range(N):
        new_point = project_points((point_cloud[i]),projection_matrix[image_number])
        if(new_point[3] > 0):
            new_point = new_point / new_point[3]
            final_points.append(new_point[0:3])
            final_colors.append(colors[i])


    print("Done : ",image_number+1, "Images out of : ", number_of_images)

final_points = np.array(final_points)
final_colors = np.array(final_colors)
mask = ((-1500 <= final_points[:,1]) & (final_points[:,1] < 1500) &
        (-1500 <= final_points[:,2]) & (final_points[:,2] < 1500) &
        (-1500 <= final_points[:,0]) & (final_points[:,0] < 1500))


final_points = final_points[mask]
print(final_points[0])
final_colors = final_colors[mask]
final_points.T[0] *= -1
print(final_points[0])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(final_points)
pcd.colors = o3d.utility.Vector3dVector(final_colors)
o3d.visualization.draw_geometries([pcd])

file_name = 'for_single_image.ply'
write_ply(file_name,np.array(final_points),np.array(final_colors*255))
