"""IVy module that manages orthorectification functions"""

import glob
from typing import Union

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import root
from shapely.affinity import rotate
from shapely.geometry import Polygon, MultiPoint
from skimage import transform
from skimage.io import imread
from tqdm import tqdm

from image_velocimetry_tools.common_functions import (
    distance,
    scale_coordinates,
    translate_coordinates,
    bounding_box_naive,
    pillow_image_to_numpy_array,
)


class CameraHelper:
    """A helper class based on the cameratransform package modified to help prepare inputs from IVy"""

    def __init__(
        self, image=None, world_points=None, image_points=None, elevation=None
    ):
        self.image_path = None
        self.image = None
        self.image_ndarray = None
        self.image_width_px = None
        self.image_height_px = None
        self.camera_matrix = None
        self.pixel_rms_error = None
        self.points_space = None
        self.points_image = None
        self.points_camera = None
        self.camera_position_world = [0, 0, 0]
        self.last_extent = None
        self.last_scaling = None
        self.map = None
        self.pixel_ground_scale_distance = None
        self.elevation = None

        if image is not None:
            self.add_image(image)
        if world_points is not None:
            self.add_space_points(world_points)
        if image_points is not None:
            self.add_image_points(image_points)
        if elevation is not None:
            if not isinstance(elevation, float):
                print("Warning: Elevation is not a float.")
            else:
                self.elevation = elevation

    def add_image_from_file(self, image_path):
        """Add image (from a file path) as a PIL Image to Class instance"""
        self.image = Image.open(image_path)
        self.image_ndarray = pillow_image_to_numpy_array(self.image)
        self.image_width_px, self.image_height_px = (
            self.image.width,
            self.image.height,
        )

    def add_image(self, image: np.ndarray):
        """Add image ndarray to Class instance"""
        self.image = image
        self.image_ndarray = self.image
        self.image_height_px, self.image_width_px, _ = self.image.shape

    def add_space_points(self, points):
        """Add space coordinate points to Class instance"""
        # ensure that the points are provided as an array
        self.points_space = np.array(points)

    def add_image_points(self, points):
        """Add image coordinate points to Class instance"""
        # ensure that the points are provided as an array
        self.points_image = np.array(points)

    def get_camera_matrix(self, Z=1.0):
        """
        Compute the projective camera matrix given point correspondences
        using the standard DLT algorithm.

        :type Z: float
        Unused here; kept only for optional compatibility.

        :returns: P (3x4 np.ndarray), rmse_r (float)
        """

        N = self.points_space.shape[0]

        # Normalize the points to improve DLT stability
        norm_img_points, T_img = normalize_points(self.points_image)
        norm_wrd_points, T_wrd = normalize_points(self.points_space)

        x = norm_wrd_points[:, 0]
        y = norm_wrd_points[:, 1]
        z = norm_wrd_points[:, 2]
        u = norm_img_points[:, 0]
        v = norm_img_points[:, 1]

        # Construct the A matrix without the Z multiplier
        A = []
        for p in range(N):
            A.append(
                [
                    x[p],
                    y[p],
                    z[p],
                    1,
                    0,
                    0,
                    0,
                    0,
                    -u[p] * x[p],
                    -u[p] * y[p],
                    -u[p] * z[p],
                    -u[p],
                ]
            )
            A.append(
                [
                    0,
                    0,
                    0,
                    0,
                    x[p],
                    y[p],
                    z[p],
                    1,
                    -v[p] * x[p],
                    -v[p] * y[p],
                    -v[p] * z[p],
                    -v[p],
                ]
            )
        A = np.asarray(A)

        # Solve via SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        P_norm = Vt[-1, :].reshape(3, 4)

        # De-normalize
        P = np.linalg.pinv(T_img) @ P_norm @ T_wrd
        P = P / P[-1, -1]  # normalize so bottom-right is 1

        self.camera_matrix = P

        # Compute reprojection error (use elevation if available)
        if self.elevation is not None:
            rmse_r, rmse_x, rmse_y = self.get_horizontal_reprojection_error(
                Z=self.elevation
            )
        else:
            rmse_r, rmse_x, rmse_y = self.get_horizontal_reprojection_error()

        self.pixel_rms_error = rmse_r

        return P, rmse_r

    def get_horizontal_reprojection_error(self, Z=0.0):
        """Compute horizontal reprojection error"""
        x = self.points_image[:, 0]
        y = self.points_image[:, 1]
        X = self.points_space[:, 0]
        Y = self.points_space[:, 1]
        points = np.array([x, y]).T
        # p, cam_pos = self.image_to_space(points, Z=Z)
        p, cam_pos = image_to_space(points, self.camera_matrix, Z=Z)
        Xr, Yr = p[:, 0], p[:, 1]
        x_diff = np.array([Xr[p] - X[p] for p in range(X.shape[0])])
        y_diff = np.array([Yr[p] - Y[p] for p in range(Y.shape[0])])
        xy_r = np.sqrt(x_diff**2 + y_diff**2)
        rmse_x = np.sqrt((np.nansum(x_diff) ** 2) / X.shape[0])
        rmse_y = np.sqrt((np.nansum(y_diff) ** 2) / Y.shape[0])
        rmse_r = np.sqrt(rmse_x**2 + rmse_y**2)

        self.camera_position_world = cam_pos
        return rmse_r, rmse_x, rmse_y

    def set_camera_matrix(self, camera_matrix):
        """Set the camera matrix

        Args:
            camera_matrix (ndarray): the camera matrix to apply
        """
        self.camera_matrix = camera_matrix
        self.pixel_rms_error = np.nan

    def __get_image_border(self, resolution=1):
        """
        Get the border of the image in a top view. Useful for drawing the field of view of the camera in a map.

        Parameters
        ----------
        resolution : number, optional
            the pixel distance between neighbouring points.

        Returns
        -------
        border : ndarray
            the border of the image in **space** coordinates, dimensions (Nx3)

        Notes
        -----
        From cameratransform package
        """
        w, h = self.image_width_px, self.image_height_px
        border = []
        for y in np.arange(0, h, resolution):
            border.append([0, y])
        for x in np.arange(0, w, resolution):
            border.append([x, h])
        for y in np.arange(h, 0, -resolution):
            border.append([w, y])
        for x in np.arange(w, 0, -resolution):
            border.append([x, 0])
        return np.array(border)

    def __get_map(self, extent=None, scaling=None, Z=0):
        """

        Parameters
        ----------
        extent
        scaling
        Z

        Returns
        -------

        Notes
        -----
        From cameratransform package
        """

        # if no extent is given, take the maximum extent from the image border
        if extent is None:
            border = self.__get_image_border()
            extent = [
                np.nanmin(border[:, 0]),
                np.nanmax(border[:, 0]),
                np.nanmin(border[:, 1]),
                np.nanmax(border[:, 1]),
            ]

        # if we have cached the map, use the cached map
        if (
            self.map is not None
            and all(self.last_extent == np.array(extent))
            and (self.last_scaling == scaling)
        ):
            return self.map

        # if no scaling is given, scale so that the resulting image has an
        # equal amount of pixels as the original image
        if scaling is None:
            scaling = np.sqrt(
                (extent[1] - extent[0]) * (extent[3] - extent[2])
            ) / np.sqrt((self.image_width_px * self.image_height_px))

        # get a mesh grid
        mesh = np.array(
            np.meshgrid(
                np.arange(extent[0], extent[1], scaling),
                np.arange(extent[2], extent[3], scaling),
            )
        )
        # convert it to a list of points Nx2
        mesh_points = mesh.reshape(2, mesh.shape[1] * mesh.shape[2]).T
        mesh_points = np.hstack(
            (mesh_points, Z * np.ones((mesh_points.shape[0], 1)))
        )

        # transform the space points to the image
        # mesh_points_shape = self.spaceToImage(mesh_points, camera_matrix=self.camera_matrix)
        mesh_points_h = get_homographic_coordinates_3D(mesh_points)

        mesh_points_shape = space_to_image(
            mesh_points_h, projection_mtx=self.camera_matrix
        )

        # reshape the map and cache it
        self.map = mesh_points_shape.T.reshape(mesh.shape).astype(np.float32)[
            :, ::-1, :
        ]

        self.last_extent = extent
        self.last_scaling = scaling

        # return the calculated map
        return self.map

    def __get_inverse_map(self, Z=0.0):
        """
        Generate the inverse mapping coordinates for converting points from the
        transformed image back to the original image.

        Returns
        -------
        inverse_x : ndarray
            The inverse mapping coordinates for the x-axis.
        inverse_y : ndarray
            The inverse mapping coordinates for the y-axis.
        """
        # Ensure that the map is computed
        assert (
            self.map is not None
        ), "Map must be computed before obtaining the inverse map."

        # Get the shape of the original image
        original_image_shape = (self.image_height_px, self.image_width_px)

        # Generate meshgrid for the original image coordinates
        mesh = np.array(
            np.meshgrid(
                np.arange(original_image_shape[1]),
                np.arange(original_image_shape[0]),
            )
        )
        mesh_points = mesh.reshape(2, mesh.shape[1] * mesh.shape[2]).T

        # Perform inverse transformation from image to space
        inverse_points, _ = image_to_space(
            mesh_points, self.camera_matrix, Z=Z
        )

        # Reshape the inverse points to match the original image shape
        inverse_x = inverse_points[:, 0].reshape(original_image_shape)
        inverse_y = inverse_points[:, 1].reshape(original_image_shape)

        return inverse_x, inverse_y

    def get_top_view_of_image(
        self,
        image,
        extent=None,
        scaling=None,
        do_plot=False,
        alpha=None,
        Z=0.0,
        skip_size_check=False,
    ):
        """
        Project an image to a top view projection. This will be done using a
        grid with the dimensions of the extent ([x_min, x_max, y_min, y_max])
        in meters and the scaling, giving a resolution. For convenience, the
        image can be plotted directly. The projected grid is cached, so if the
        function is called a second time with the same parameters, the second
        call will be faster.

        Parameters
        ----------
        image : ndarray
            the image as a numpy array.
        extent : list, optional
            the extent of the resulting top view in meters: [x_min, x_max,
            y_min, y_max]. If no extent is given a suitable
            extent is guessed. If a horizon is visible in the image, the
            guessed extent will in most cases be too streched.
        scaling : number, optional
            the scaling factor, how many meters is the side length of each
            pixel in the top view. If no scaling factor is given, a good
            scaling factor is guessed, trying to get about the same
            number of pixels in the top view as in the original image.
        do_plot : bool, optional
            whether to directly plot the resulting image in a matplotlib figure.
        alpha : number, optional
            an alpha value used when plotting the image. Useful if multiple
            images should be overlaid.
        Z : number, optional
            the "height" of the plane on which to project.
        skip_size_check : bool, optional
            if true, the size of the image is not checked to match the size of
            the cameras image.

        Returns
        -------
        image : ndarray
            the top view projected image

        Notes
        -----
        From cameratransform package
        """

        # check if the size of the image matches the size of the camera
        if not skip_size_check:
            assert image.shape[1] == self.image_width_px, (
                "The with of the image (%d) does not match the image width of the camera (%d)"
                % (image.shape[1], self.image_width_px)
            )
            assert image.shape[0] == self.image_height_px, (
                "The height of the image (%d) does not match the image height of the camera (%d)."
                % (image.shape[0], self.image_height_px)
            )

        # Check for existing scaling
        if self.last_scaling is not None:
            scaling = self.last_scaling

        # get the mapping
        x, y = self.__get_map(extent=extent, scaling=scaling, Z=Z)

        # Transform the image
        image = cv2.remap(
            image,
            x,
            y,
            interpolation=cv2.INTER_NEAREST,
            borderValue=[0, 1, 0, 0],
        )  # , borderMode=cv2.BORDER_TRANSPARENT)
        self.pixel_ground_scale_distance = np.mean(
            (
                np.diff(self.last_extent)[0] / image.shape[1],
                np.diff(self.last_extent)[-1] / image.shape[0],
            )
        )

        if do_plot:
            import matplotlib.pyplot as plt

            plt.imshow(image, extent=self.last_extent, alpha=alpha)

        return image

    def get_inverse_top_view_point(self, transformed_point):
        """
        Convert a point from the transformed top view image back to the original image.

        Parameters
        ----------
        transformed_point : tuple
            The coordinates of the point in the transformed image (x, y).

        Returns
        -------
        original_point : tuple
            The coordinates of the corresponding point in the original image (x, y).
        """
        # Get the inverse mapping
        inverse_x, inverse_y = self.__get_inverse_map()

        # Perform inverse mapping with interpolation
        x = np.interp(
            transformed_point[0],
            np.arange(inverse_x.shape[1]),
            inverse_x[int(transformed_point[0]), :],
        )
        y = np.interp(
            transformed_point[1],
            np.arange(inverse_y.shape[0]),
            inverse_y[:, int(transformed_point[1])],
        )
        z = 1
        new_position = space_to_image(
            np.array([[x, y, z, 1]]), self.camera_matrix
        )

        return new_position[:, 0], new_position[:, 1]

    def map_points_to_top_view(self, points, extent, scaling=None, Z=0.0):
        """
        Map points from the source image to their corresponding positions in the top-down view.

        Parameters:
        - points: ndarray
            Array of pixel coordinates in the source image. Each row represents a point in the format [x, y].
        - extent: list
            Extent of the resulting top-down view in meters: [x_min, x_max, y_min, y_max].
        - scaling: float
            Scaling factor, indicating how many meters is the side length of each pixel in the top-down view.
        - Z: float, optional
            Height of the plane on which to project.

        Returns:
        - top_view_points: ndarray
            Array of pixel coordinates in the top-down view corresponding to the input points.
        """

        # Check for existing scaling
        if self.last_scaling is not None:
            scaling = self.last_scaling

        # Generate the mapping
        x, y = np.meshgrid(
            np.linspace(extent[0], extent[1], num=points.shape[0]),
            np.linspace(extent[2], extent[3], num=points.shape[0]),
        )

        # Map points to top view
        top_view_points = np.zeros_like(points)
        for i, (px, py) in enumerate(points):
            top_view_points[i] = [
                int((px - extent[0]) / scaling),
                int((py - extent[2]) / scaling),
            ]

        return top_view_points


# def image_to_space(self, points, X=None, Y=None, Z=0):
#     # TODO: move out of class and at @jit
#     # Check that camera_matrix is 3x4 array
#     if not isinstance(self.camera_matrix, np.ndarray) or self.camera_matrix.shape != (3, 4):
#         raise ValueError("Projection matrix array is not of the right shape (3x4)")
#
#     # ensure that the points are provided as an array
#     points = np.array(points)
#     offset = np.array(self.camera_position_world)
#
#     # get the index which coordinate to force to the given value
#     given = np.array([X, Y, Z], dtype=object)
#     if X is not None:
#         index = 0
#     elif Y is not None:
#         index = 1
#     elif Z is not None:
#         index = 2
#
#     ray_directions, camera_center = compute_rays(points, self.camera_matrix)
#     if ray_directions.shape[1] > 1:
#         direction = ray_directions.T
#
#     # solve the line equation for the factor (how many times the direction vector needs to be added to the origin point)
#     factor = (given[index] - offset[..., index]) / direction[..., index]
#
#     e_tol = 1e-6
#
#     # Represent the plane and its normal, parallel to supplied constraint
#     p_on_plane = np.zeros((3,))
#     p_on_plane[index] = given[index]
#     p_normal = np.zeros((3,))
#     p_normal[index] = 1.0
#
#     ndotu = p_normal.dot(ray_directions)
#     # ndotu[ndotu < e_tol] = np.nan
#     w = camera_center - p_on_plane
#     si = -p_normal.dot(w) / ndotu
#     psi = w[:, None] + si * ray_directions + p_on_plane[:, None]
#     psi = psi.T
#
#     # if not isinstance(factor, np.ndarray):
#     #     # if factor is not an array, we don't need to specify the broadcasting
#     #     new_points = direction * factor + offset
#     # else:
#     #     # apply the factor to the direction vector plus the camera position
#     #     new_points = direction * factor[:, None] + camera_center[None, :]
#     #     # ignore points that are behind the camera (e.g. trying to project points above the horizon to the ground)
#     # points[factor < 0] = np.nan
#
#     return psi, camera_center


class FourPointSolution:
    """
    Solve for quadrilateral Cartesian coordinates given distances between vertices

    Parameters
    ----------
    point_distances : array-like, optional
        Distances between the vertices of the quadrilateral. If not provided, default distances are set to zero.
    z : float, optional
        The z-coordinate for all vertices. Defaults to 0.0, assuming a level water surface.

    Attributes
    ----------
    p : float
        Distance between vertices P and Q.
    q : float
        Distance between vertices P and S.
    r : float
        Distance between vertices Q and S.
    s : float
        Distance between vertices Q and R.
    t : float
        Distance between vertices P and R.
    u : float
        Distance between the intersection of diagonals and vertex P.
    solution : OptimizeResult or None
        Result of the optimization solver.
    x : ndarray, shape (4,)
        x-coordinates of the vertices.
    y : ndarray, shape (4,)
        y-coordinates of the vertices.
    z : ndarray, shape (4,)
        z-coordinates of the vertices.

    Methods
    -------
    get_world_coordinates(extra=False)
        Return the world coordinates of the vertices.
    set_distances(arguments)
        Set known distances for sides and diagonals of the quadrilateral.
    quadrilateral(v)
        System of equations to optimize for finding the coordinates.
    solve()
        Optimization solver to estimate unknowns given the quadrilateral equations.

    Notes
    -----
    The class uses an optimization solver to find the coordinates of a quadrilateral given the distances between its vertices.
    - The distance between P and R (t) should represent a line parallel to the x-axis.
    - The distance between Q and S (s) should represent a line more or less aligned with the y-axis.
    - The distance between P and S (q) should represent a line more or less aligned with the y-axis.
    - The distance between Q and R (r) may not be exactly parallel but should be nearly parallel to the x-axis.
    - The distance between the intersection of diagonals and vertex P (u) should ensure a positive slope for the line from P to R.

    Examples
    --------
    # Distances between quadrilateral PQRS and diagonals (PR, SQ)
    # for the Unnamed Creek site
    roi_distances = [19, 13.5, 21.5, 21.5, 23.5, 30]
    worldCoordinatesROI = FourPointSolution(roi_distances).get_world_coordinates()
    print(worldCoordinatesROI)
    >>> [[ 0.          0.          0.        ]
        [19.          0.          0.        ]
        [18.66567008 13.97160873  0.        ]
        [-1.64971249 21.56672733  0.        ]]
    """

    def __init__(self, point_distances=None, z=0.0):
        if point_distances is not None:
            p, q, r, s, t, u = point_distances
        else:
            p, q, r, s, t, u = np.zeros(6)
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.t = t
        self.u = u
        self.v = [1, 1, 1, 1]
        self.solution = None
        self.x = None
        self.y = None
        self.z = np.array([z, z, z, z])  # Assume level water surface
        self.solve()

    def get_world_coordinates(self, extra=False):
        """Return world coordinates"""
        x = self.x
        y = self.y
        z = self.z
        vertices = np.array(
            [
                (x[0], y[0], z[0]),
                (x[1], y[1], z[1]),
                (x[2], y[2], z[2]),
                (x[3], y[3], z[3]),
            ]
        )

        # Create a few more points by rotating the ROI by 90 degs,
        # and keeping the centroid
        polygon = Polygon(MultiPoint(vertices))
        xc, yc = polygon.centroid.xy
        polygon2 = rotate(polygon, 90)
        xx, yy = polygon2.exterior.coords.xy
        extra_points = np.vstack(
            (np.array(tuple(zip(xx[:-1], yy[:-1]))), np.array([xc, yc]).T)
        )
        extra_points = np.hstack(
            (extra_points, np.ones((extra_points.shape[0], 1)) * z[0])
        )
        extra_points = np.vstack((vertices, extra_points))
        if extra:
            return vertices, extra_points
        else:
            return vertices

    def set_distances(self, arguments):
        """Set known distances for sides and diagonals of quadrilateral"""
        p = arguments[0]
        q = arguments[1]
        r = arguments[2]
        s = arguments[3]
        t = arguments[4]
        u = arguments[5]
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.t = t
        self.u = u

    def quadrilateral(self, v):
        """System of equations to optimize"""
        # Set known values
        p = self.p
        q = self.q
        r = self.r
        s = self.s
        t = self.t
        u = self.u

        # System of equations to optimize
        f = [
            v[0] ** 2 + v[1] ** 2 - t**2,  # xR
            v[2] ** 2 + v[3] ** 2 - s**2,  # yR
            (v[0] - p) ** 2 + v[1] ** 2 - q**2,  # xS
            (v[0] - v[2]) ** 2 + (v[1] - v[3]) ** 2 - r**2,  # yS
            np.abs(
                ((p - v[2]) ** 2 + v[3] ** 2 - u**2)
            ),  # xI, yI  (intersection of diagonals)
        ]
        return f

    def solve(self):
        """Optimization solver to estimate unknowns given the quadrilateral equations"""
        # Perform optimization with constraints
        solution = root(
            self.quadrilateral, np.array([1, 1, 1, 1]), method="lm"
        )

        # Ensure p is parallel to x-axis
        self.x = np.array([0, self.p, solution.x[0], solution.x[2]])
        self.y = np.array([0, 0, solution.x[1], solution.x[3]])

        # # Ensure r is nearly parallel (adjust as needed)
        # # You can adjust the slope threshold depending on your requirements
        # slope_threshold = 0.01  # Adjust as needed
        # if abs((self.y[2] - self.y[3]) / (
        #         self.x[2] - self.x[3])) < slope_threshold:
        #     # If slope is too small, adjust the point
        #     self.x[3] += 0.1  # Adjust as needed
        #     self.y[3] += 0.1  # Adjust as needed
        #
        # # Ensure s and q are more or less aligned with y-axis
        # # No additional action needed, as these are already ensured by the equations
        #
        # # Ensure t has a positive slope
        # if self.x[2] < self.x[3]:
        #     # If x2 < x3, swap the points
        #     self.x[2], self.x[3] = self.x[3], self.x[2]
        #     self.y[2], self.y[3] = self.y[3], self.y[2]

        # Update the solution attribute
        self.solution = solution
        return solution


def find_homography_matrix(source_points, destination_points):
    """Estimate homography given point correspondences."""
    # Convert points to numpy arrays
    source_points = np.array(source_points)
    destination_points = np.array(destination_points)

    # Ensure points have the correct shape
    if source_points.shape[1] != 2 or destination_points.shape[1] != 2:
        raise ValueError("Point arrays must have shape (N, 2)")

    homography_matrix, status = cv2.findHomography(
        source_points, destination_points
    )

    # submatrix = homography_matrix[:2, :2]
    # determinant = np.linalg.det(submatrix)
    #
    # # Check for mirroring and flip the points if necessary
    # if determinant < 0:
    #     logging.info(f"HOMOGRAPHY: The homography matrix indicated image may "
    #                  f"be mirrored. Check the result.")
    #     source_points = source_points[::-1]
    #
    #     homography_matrix, status = cv2.findHomography(
    #         source_points, destination_points
    #     )
    return homography_matrix


def warp_image_homography(
    image, homography_matrix, output_shape=(200, 200), clip=True
) -> np.ndarray:
    """Warp input image using homography matrix and return warped image as a ndarray.

    :param image:
    :param homography_matrix: np.ndarray
    :param output_shape: (height, width) tuple
    :param clip: bool
    :return:
    """
    inverse_homography = np.linalg.inv(homography_matrix)
    return transform.warp(
        image, inverse_homography, output_shape=output_shape, clip=clip
    )


def rectify_homography(
    image,
    points_world_coordinates,
    points_perspective_image_coordinates,
    homography_matrix=None,
    transformed_roi=None,
    pad_x=200,
    pad_y=200,
):
    """Rectify an image given supplied world and pixel point correspondences"""
    # TODO: this rectification function is slow. Improve processing for speed. (Issue #4)
    #      cProfile shows the slow down is skimage's warp. Investigate using openCV warp instead.
    #      e.g. https://pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

    # Compute world and image scale, using the diagonal of the ROI quadrilateral,
    # then scale the world_points so that they "fit" into a box the same size as
    # the input image.
    world_scale = distance(
        points_world_coordinates[0], points_world_coordinates[2]
    )
    original_world_bounding_box = bounding_box_naive(points_world_coordinates)
    padding = 0.0  # percent padding per side
    scale_factor = image.shape[1] / (
        (original_world_bounding_box[1][0] - original_world_bounding_box[0][0])
        * (1 + padding * 2)
    )
    scaled_destination_image_points = scale_coordinates(
        points_world_coordinates, scale_factor
    )
    min_x = np.min([p[0] for p in scaled_destination_image_points])
    min_y = np.min([p[1] for p in scaled_destination_image_points])
    # max_x = np.max([p[0] for p in scaled_world_coordinates])
    # max_y = np.max([p[1] for p in scaled_world_coordinates])

    # Translate the scaled world_points so that they fit entirely into the image (all positive coords) and given user
    # request padding in x and y, then recompute the bounding box
    cx = (
        np.abs(0 - min_x)
        + (
            image.shape[1]
            - np.max([p[0] for p in scaled_destination_image_points])
        )
        / 2
    )
    cy = np.abs(0 - min_y)
    scaled_destination_image_points = translate_coordinates(
        scaled_destination_image_points, cx + pad_x / 2, cy + pad_y / 2
    )
    bbox = bounding_box_naive(scaled_destination_image_points)

    # Generate the canvas for the new perspective warped image. Allow some padding so that the entire
    # ROI will definitely be in the canvas
    dst_size = np.ceil(
        np.max(
            [
                image.shape[1] + pad_x * 2,
                np.ceil((bbox[1][1] - bbox[0][1])) + pad_y * 2,
            ]
        )
    )
    width = dst_size
    height = dst_size

    # Transform the image, and compute the pixel Ground Scale Distance
    # If the homography and transformed ROI are already calculated, we don't need to do it again.
    if homography_matrix is None and transformed_roi is None:
        homography_matrix = find_homography_matrix(
            points_perspective_image_coordinates,
            scaled_destination_image_points,
        )
    # transformed_image = warp_image_homography(image, homography_matrix, (height, width), clip=True)
    transformed_image = cv2.warpPerspective(
        image, homography_matrix, (int(width), int(height))
    )
    # transformed_image = np.flip(transformed_image, axis=-1)

    if transformed_roi is None:  # User did not provide an ROI, so create one
        x_dst = [val[0] for val in scaled_destination_image_points] + [
            scaled_destination_image_points[0][0]
        ]
        y_dst = [val[1] for val in scaled_destination_image_points] + [
            scaled_destination_image_points[0][1]
        ]
        transformed_roi = (x_dst, y_dst)

    # transformedBBox = bounding_box_naive(transformed_roi)
    transformed_pixel_scale = distance(
        (transformed_roi[0][0], transformed_roi[1][0]),
        (transformed_roi[0][2], transformed_roi[1][2]),
    )
    pixel_gsd = world_scale / transformed_pixel_scale

    # Ensure transformed_image is scaled as an uint8 image
    # info = np.iinfo(transformed_image.dtype)  # Get the information of the incoming image type
    # transformed_image = transformed_image.astype(np.float64) / info.max  # normalize the data to 0 - 1
    # transformed_image = 255 * transformed_image  # Now scale by 255
    # transformed_image = transformed_image.astype(np.uint8)

    # Convert everything to ndarrays
    transformed_roi = np.array(transformed_roi)
    scaled_destination_image_points = np.array(scaled_destination_image_points)
    scaled_destination_image_points = scaled_destination_image_points[
        ..., np.newaxis
    ]  # Forces shape (x,y,1)

    return (
        transformed_image,
        transformed_roi,
        scaled_destination_image_points,
        pixel_gsd,
        homography_matrix,
    )


# @njit()
def rectify_many_homography(
    images,
    world_points,
    image_points,
    start_frame=None,
    save_output=True,
    output_location=None,
):
    """Driver script to rectify and save outputs from a frame sequence"""
    # For naming file, if no start frame is specified, just name image
    # sequence starting at 1
    if start_frame is None:
        start_frame = 1

    # Initial call to print 0% progress
    num_frames = len(images)
    print(
        "Writing {} transformed image frames to {}".format(
            num_frames, output_location
        )
    )

    count = 0
    transformed_images = []
    for image in tqdm(images, total=num_frames):
        t, transformed_roi, pixel_gsd = rectify_homography(
            image, world_points, image_points
        )
        transformed_images.append(t)
        if save_output:
            img = Image.fromarray(t)
            img.save(
                output_location + "/t{:05d}.jpg".format(start_frame + count)
            )
        count += 1
    return (
        transformed_images,
        transformed_roi,
    )


def transform_points_with_homography(points, H):
    """
    Transforms 2D points using homography matrix H.

    Arguments:
    points : numpy.ndarray
        Array of 2D points with shape (N, 2).
    H : numpy.ndarray
        Homography matrix.

    Returns:
    numpy.ndarray
        Transformed points with shape (N, 2).
    """
    # Reshape points to match OpenCV's requirements (Nx1x2)
    points_reshaped = points.reshape(-1, 1, 2)

    # Perform perspective transformation
    transformed_points = cv2.perspectiveTransform(points_reshaped, H)

    # Reshape transformed points to match the original shape
    transformed_points = transformed_points.reshape(-1, 2)

    return transformed_points


def compute_homography_matrix_from_camera_matrix(
    K, world_points, original_points, wse
):
    """
    Computes the homography matrix that represents a plane at a given elevation in the image.

    Arguments:
    K : numpy.ndarray
        Camera matrix.
    world_points : numpy.ndarray
        Array of real-world points with shape (N, 3).
    original_points : numpy.ndarray
        Array of pixel coordinates of the points with shape (N, 2).
    wse : float
        Elevation of the plane in the image.

    Returns:
    numpy.ndarray
        Homography matrix.
    """
    # Step 1: Calculate the transformation matrix M
    # M, _ = cv2.findHomography(world_points, original_points)
    M, _ = cv2.findHomography(world_points[:, :2], original_points)

    # Step 2: Adjust M for the given elevation
    M[:, 2] /= wse

    # Step 3: Compute the homography matrix H using adjusted transformation matrix
    H = np.dot(K, M)

    return H


def cartesian_to_homogeneous(cartesian_array: np.ndarray) -> np.ndarray:
    """Convert numpy array of points from Cartesian to Homogeneous coordinates using broadcasting.

    Notes
    -----
    Adapted from: https://codereview.stackexchange.com/q/257349
    """
    if isinstance(cartesian_array, tuple):
        cartesian_array = np.array(cartesian_array)
    shape = cartesian_array.shape
    homogeneous_array = np.ones((*shape[:-1], shape[-1] + 1))
    homogeneous_array[..., :-1] = cartesian_array
    return homogeneous_array


def homogeneous_to_cartesian(homogeneous_array: np.ndarray) -> np.ndarray:
    """Convert numpy array of points from Homogeneous to Cartesian coordinates using broadcasting.

    Notes
    -----
    Adapted from: https://codereview.stackexchange.com/q/257349
    """
    if isinstance(homogeneous_array, tuple):
        homogeneous_array = np.array(homogeneous_array)
    return homogeneous_array[..., :-1] / homogeneous_array[..., [-1]]


# Numba does not support einsum, but this is much faster without jit using einsum
# @jit(nopython=True, signature_or_function='f8[:,:](f8[:,:],f8[:,:])')
def space_to_image(points, projection_mtx):
    """Convert space coordinates into image coordinates given a Projection matrix

    Parameters
    ----------
    points : np.float64
        4xn ndarray of X,Y,Z points in homogeneous coordinates [X, Y, Z, 1]
    projection_mtx : np.float64
        3x4 ndarray representing the Projection Matrix

    Returns
    -------
    new_points : np.float64
        2xn ndarray of image coordinates corresponding to input space
        coordinate points
    """

    # Test inputs
    assert projection_mtx.shape == (
        3,
        4,
    ), f"Projection matrix array is not of the right shape (3x4)"

    # Convert input points using the camera matrix. This approach uses np.einsum to perform
    # the matmul of each point (as a row) and the projection matrix. It is equivalent to
    #       pp = projection_mtx @ p, along axis=1
    #
    # Highly optimized performance is faster than matmul, including with @jit
    new_points = np.einsum("ij,kj->ki", projection_mtx, points)
    new_points = np.vstack(
        (
            new_points[:, 0] / new_points[:, -1],
            new_points[:, 1] / new_points[:, -1],
        )
    ).T
    return new_points


# @jit(nopython=True, signature_or_function='Tuple((f8[:,:], f8[:,:]))(f8[:,:],f8[:,:],f8)')
def image_to_space(points, camera_matrix, X=None, Y=None, Z=0.0):
    """Convert image coordinates to space coordinates

    Args:
        points (ndarray): the points to convert
        camera_matrix (ndarray): the camera matrix
        X (float, optional): constrain which plane to transform. Defaults to None.
        Y (float, optional): constrain which plane to transform. Defaults to None.
        Z (float, optional): constrain which plane to transform. Defaults to 0.0.

    Returns:
        tuple: tuple containing the transformed points and the new camera center location
    """
    # Check that camera_matrix is 3x4 array
    if not isinstance(camera_matrix, np.ndarray) or camera_matrix.shape != (
        3,
        4,
    ):
        raise ValueError(
            "Projection matrix array is not of the right shape (3x4)"
        )

    # ensure that the points are provided as an array
    points = np.array(points)
    offset = np.array([0, 0, 0])

    # get the index which coordinate to force to the given value
    given = np.array([X, Y, Z], dtype=object)
    if X is not None:
        index = 0
    elif Y is not None:
        index = 1
    elif Z is not None:
        index = 2

    ray_directions, camera_center = compute_rays(points, camera_matrix)
    if ray_directions.shape[1] > 1:
        direction = ray_directions.T

    # solve the line equation for the factor (how many times the direction vector needs to be added to the origin point)
    factor = (given[index] - offset[..., index]) / direction[..., index]

    e_tol = 1e-6

    # Represent the plane and its normal, parallel to supplied constraint
    p_on_plane = np.zeros((3,))
    p_on_plane[index] = given[index]
    p_normal = np.zeros((3,))
    p_normal[index] = 1.0

    ndotu = p_normal.dot(ray_directions)
    # ndotu[ndotu < e_tol] = np.nan
    w = camera_center - p_on_plane
    si = -p_normal.dot(w) / ndotu
    psi = w[:, None] + si * ray_directions + p_on_plane[:, None]
    new_points = psi.T

    # if not isinstance(factor, np.ndarray):
    #     # if factor is not an array, we don't need to specify the broadcasting
    #     new_points = direction * factor + offset
    # else:
    #     # apply the factor to the direction vector plus the camera position
    #     new_points = direction * factor[:, None] + camera_center[None, :]
    #     # ignore points that are behind the camera (e.g. trying to project points above the horizon to the ground)
    # points[factor < 0] = np.nan

    return new_points, camera_center


def projective_to_conventional(points_mx3: np.ndarray):
    """Convert projective coordinates into conventional coordinates."""

    assert points_mx3.shape[-1] == 3

    new_points = np.vstack(
        (
            points_mx3[:, 0] / points_mx3[:, -1],
            points_mx3[:, 1] / points_mx3[:, -1],
        )
    ).T
    return new_points


def space_to_camera(points, P):
    """Convert points in the space coordinates to camera coordinates."""
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    points = np.array(points)
    return np.dot(points - T, R.T)


def get_homographic_coordinates_2D(points_nx2: np.ndarray) -> np.ndarray:
    """Convert 2D points into homogenous coordinates."""
    shp = points_nx2.shape
    if shp == (2,):
        return np.hstack((points_nx2, 1)).T
    else:
        return (np.hstack([points_nx2, np.ones((points_nx2.shape[0], 1))])).T


def get_homographic_coordinates_3D(points_nx3: np.ndarray) -> np.ndarray:
    """Convert 3D points into homogenous coordinates."""
    shp = points_nx3.shape
    if shp == (3,):
        return np.hstack((points_nx3, 1)).T
    else:
        return np.hstack([points_nx3, np.ones((points_nx3.shape[0], 1))])


def normalize_points(points: np.ndarray) -> Union[np.ndarray, np.ndarray]:
    """Homogenous normalization of points."""
    dims = points.shape[1]
    m, s = np.mean(points, axis=0), np.std(points)
    if dims == 2:
        T = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        T = np.array(
            [[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]]
        )
    T = np.linalg.inv(T)
    points = T @ np.concatenate((points.T, np.ones((1, points.shape[0]))))
    points = points[0:dims, :].T
    return points, T


# @jit(nopython=True, signature_or_function='Tuple((f8[:,:], f8[:,:]))(f8[:,:],f8[:,:])')
def compute_rays(
    points: np.ndarray, camera_matrix: np.ndarray
) -> Union[np.ndarray, np.ndarray]:
    """Compute rays which pass through supplied points.

    Notes
    -----
    see Harley & Zisserman pg 162, section 6.2.2, figure 6.14
    see https://math.stackexchange.com/a/597489/541203
    """
    # ensure that the points are provided as an array
    points = np.array(points)

    # Check that camera_matrix is 3x4 array
    if not isinstance(camera_matrix, np.ndarray) or camera_matrix.shape != (
        3,
        4,
    ):
        raise ValueError(
            "Projection matrix array is not of the right shape (3x4)"
        )

    m_3x3 = camera_matrix[:, :3]
    p4_3x1 = camera_matrix[:, 3]
    m_inv_3x3 = np.linalg.inv(m_3x3)

    # projection matrix to camera center
    camera_center_3x1 = np.expand_dims(-m_inv_3x3 @ p4_3x1, 1)

    # projection matrix + pixel locations to ray directions
    points_sxn = get_homographic_coordinates_2D(points)
    ray_directions_3xn = m_inv_3x3 @ points_sxn
    return ray_directions_3xn, camera_center_3x1.reshape(-1)


def rq_decomposition(matrix: np.ndarray):
    """Perform RQ Decomposition on supplied array.

    :param matrix:
    :return:
    """
    from scipy.linalg import qr

    q, r = qr(np.flipud(matrix).T)
    r = np.flipud(r.T)
    q = q.T
    return r[:, ::-1], q[::-1, :]


# def projective_matrix_to_camera_matrix(P: np.ndarray):
#     """Use QR factorization to extract the camera intrinsics from a projective matrix.
#
#     Notes
#     -----
#     See Szelinski equations 2.56 & 2.57 and associated text.
#     Code implementation from J. E. Solem: http://www.janeriksolem.net/2011/03/rq-factorization-of-camera-matrices.html
#     """
#
#     assert P.shape == (3, 4)
#     # factor the first 3x3 of the projective matrix
#     K, R = rq_decomposition(P[:, :3])
#
#     # make diagonal of K positive
#     T = np.diag(np.sign(np.diag(K)))
#
#     K = np.dot(K, T)
#     R = np.dot(T, R)  # T is its own inverse
#
#     return K, R


def projective_matrix_to_camera_matrix(P: np.ndarray):
    """Use QR factorization to extract the camera intrinsics from a projective matrix."""
    assert P.shape == (3, 4)

    # Factor the first 3x3 of the projective matrix
    K, R = rq_decomposition(P[:, :3])

    # Make diagonal of K positive and normalize it
    K = np.dot(K, np.diag(1 / np.abs(np.diag(K))))

    T = np.diag(np.sign(np.diag(K)))

    # Make sure the scaling factors are positive
    R = np.dot(np.diag(np.sign(np.diag(K))), R)

    return K, R


def rectify_camera(
    images, outputFolder, camera_matrix, extent=None, Z=0, start_frame=0
):
    """Use the CameraHelper class to rectify a single image

    Args:
        images (list): the images to transforms. Note this function takes the first item in the list
        outputFolder (str): a path to the output folder location
        camera_matrix (ndarray): the camera matrix
        extent (ndarray, optional): an array describing the image transformation extent. This is a bounding box over which the image will be trasformed. Defaults to None.
        Z (float, optional): the z-plane to use. Defaults to 0.
        start_frame (int, optional): the starting frame number. Defaults to 0.

    Returns:
        object: CameraHelper object
    """
    count = 0
    camera = CameraHelper(image_path=images[0])
    camera.set_camera_matrix(camera_matrix)
    for i in images:
        t = camera.get_top_view_of_image(imread(i), extent=extent, Z=Z)
        img = Image.fromarray(t)
        img.save(outputFolder + "/t{:05d}.jpg".format(start_frame + count))
        count += 1
    return camera


def rectify_many_camera(batchConfigList, extent=None, startFrame=0):
    """Use the CameraHelper class to rectify multiple images

    Args:
        batchConfigList (list): batch configuration list
        extent (ndarray, optional): an array describing the image transformation extent. This is a bounding box over which the image will be trasformed. Defaults to None
        startFrame (int, optional): the index of the first frame to transform. Defaults to 0.

    Returns:
        _type_: _description_
    """
    imagesFolder = batchConfigList[0]
    Z = batchConfigList[1]
    projection_mtx = batchConfigList[2]
    extent = batchConfigList[3]
    images = sorted(glob.glob(imagesFolder + "/f*.jpg"))
    count = 0
    # Load first image as numpy array for CameraHelper initialization
    first_image = imread(images[0])
    camera = CameraHelper(image=first_image)
    camera.set_camera_matrix(projection_mtx)
    for i in images:
        t = camera.get_top_view_of_image(imread(i), extent=extent, Z=Z)
        img = Image.fromarray(t)
        img.save(imagesFolder + "/t{:05d}.jpg".format(startFrame + count))
        count += 1
    return 0


def calculate_homography_matrix_simple(point_pairs):
    """
    Calculate the homography matrix using the Direct Linear Transform (DLT) method.

    Args:
        point_pairs: A list of tuples containing the correspondence points [(x1, y1, X1, Y1), (x2, y2, X2, Y2)].

    Returns:
        H: The homography matrix.
    """
    if len(point_pairs) < 2:
        raise ValueError("At least 2 point correspondences are required.")

    A = []
    for x, y, X, Y in point_pairs:
        A.append([-X, -Y, -1, 0, 0, 0, x * X, x * Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])

    A = np.array(A)

    # Solve for the homography matrix using Singular Value Decomposition (SVD).
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)

    return H


def estimate_view_angle(matrix):
    """
    Estimate off-nadir viewing angle (in degrees) from either a homography matrix (3x3)
    or a full camera projection matrix (3x4 or 4x3).

    Parameters
    ----------
    matrix : np.ndarray
        A 3x3 homography matrix (H) or a 3x4 / 4x3 camera matrix (P).

    Returns
    -------
    view_angle_deg : float
        Approximate off-nadir viewing angle in degrees. 0° = nadir.
    """

    matrix = np.asarray(matrix)

    if matrix.shape == (3, 3):
        # Homography matrix
        H = matrix / np.linalg.norm(matrix[:, 0])  # normalize to remove scale
        h1 = H[:, 0]
        h2 = H[:, 1]
        # Estimate angle between warped axes
        cos_theta = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        view_angle_deg = abs(np.degrees(theta_rad) - 90.0)
        return view_angle_deg

    elif matrix.shape in [(3, 4), (4, 3)]:
        # Camera projection matrix P = K [R | t]
        if matrix.shape == (4, 3):
            matrix = matrix.T  # Convert to 3x4
        R_t = matrix[:, :3]  # ignore translation
        # Third column of R is viewing direction in camera space
        view_dir = R_t[2, :]  # row vector
        # Assume nadir is aligned with world Z axis [0, 0, 1]
        nadir = np.array([0, 0, 1])
        cos_angle = np.dot(view_dir, nadir) / np.linalg.norm(view_dir)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        view_angle_deg = np.degrees(angle_rad)
        return view_angle_deg

    else:
        raise ValueError(
            "Input must be a 3x3 homography or 3x4 / 4x3 camera matrix."
        )


def estimate_orthorectification_rmse(view_angle_deg, gsd_m):
    """
    Estimate scene-wide orthorectification RMSE (in meters) from view angle
    and GSD.

    This method makes simple assumptions about digitization errors. Specific
    assumptions include:
    - Pixel-level reprojection errors grow as view angle increases from
      nadir (0°).
    - Arbitrary base error of 1.5–3 pixels at nadir
    - Add angular-dependent factor to account for obliqueness distortion
    - View angle beyond 45° leads to significant parallax and
      projective errors.

    Parameters
    ----------
    view_angle_deg : float
        Estimated off-nadir viewing angle in degrees (0° = nadir).
    gsd_m : float
        Ground sample distance in meters per pixel.

    Returns
    -------
    rmse_m : float
        Estimated orthorectification RMSE in meters.
    """

    angle = abs(view_angle_deg)

    # Estimate pixel RMSE based on angle ranges
    if angle <= 5:
        pixel_rmse = 1.5
    elif angle <= 15:
        pixel_rmse = 2.0
    elif angle <= 30:
        pixel_rmse = 3.0
    elif angle <= 45:
        pixel_rmse = 4.5
    else:
        pixel_rmse = 6.0

    # Scale by ground resolution
    rmse_m = pixel_rmse * gsd_m
    return rmse_m


def estimate_scale_based_rmse(gsd_m, baseline_m, pixel_error_per_point=2.0):
    """
    Estimate orthorectification RMSE (in meters) for scale-based mapping.

    Assumes the user digitizes two points to define a known real-world
    distance. Also, this method neglects errors associated with camera
    distortion.

    Parameters
    ----------
    gsd_m : float
        Ground sample distance (meters per pixel).
    baseline_m : float
        Real-world distance between the two scale points (in meters).
    pixel_error_per_point : float, optional
        Assumed digitizing error per point (pixels). Default is 2.0 pixels.

    Returns
    -------
    rmse_m : float
        Estimated orthorectification RMSE in meters.
    """

    import numpy as np

    # Total pixel error across both points
    total_pixel_error = np.sqrt(
        2 * pixel_error_per_point**2
    )  # Pythagorean error

    # Estimated relative error in scale
    relative_scale_error = (total_pixel_error * gsd_m) / baseline_m

    # Apply that relative error across the scene (worst case: full error applies)
    # Assume average pixel reprojection error is similar to pixel_error_per_point
    rmse_m = relative_scale_error * baseline_m

    return rmse_m


def pixels_to_world(pixel_coords, H):
    """
    Convert pixel coordinates to world coordinates using the homography matrix H.

    Args:
        pixel_coords: A list of pixel coordinates as 2D arrays [[x1, y1], [x2, y2], ...].
        H: The homography matrix.

    Returns:
        world_coords: A NumPy array of the same shape as pixel_coords containing corresponding world coordinates.
    """
    world_coords = np.zeros_like(pixel_coords, dtype=float)

    # Calculate the inverse of the homography matrix
    H_inv = np.linalg.inv(H)

    for i, pixel_coord in enumerate(pixel_coords):
        # Add a homogeneous coordinate (1) to the pixel coordinate
        pixel_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1])

        # Apply the inverse homography transformation
        world_homogeneous = np.dot(H_inv, pixel_homogeneous)

        # Normalize the result to obtain the world coordinate
        world_coord = world_homogeneous / world_homogeneous[-1]

        world_coords[i] = world_coord[
            :2
        ]  # Discard the last element (homogeneous coordinate)

    return world_coords


def compute_translation_and_scale(world_coords, pixel_coords):
    """
    Compute translation and scale factors to transform pixel coordinates to world coordinates.

    Args:
        world_coords: A list of world coordinates as 2D arrays [[X1, Y1], [X2, Y2], ...].
        pixel_coords: A list of pixel coordinates as 2D arrays [[x1, y1], [x2, y2], ...].

    Returns:
        translation: A 2D array [tx, ty] representing the translation.
        scale: A 2D array [sx, sy] representing the scale factors.
    """
    # Calculate the translation by finding the difference between the means
    # of world and pixel coordinates
    mean_world_coords = np.mean(world_coords, axis=0)
    mean_pixel_coords = np.mean(pixel_coords, axis=0)
    translation = mean_world_coords - mean_pixel_coords

    # Calculate the scale factors by finding the ratios of the means of
    # world and pixel coordinates
    mean_world_dist = np.mean(
        np.linalg.norm(world_coords - mean_world_coords, axis=1)
    )
    mean_pixel_dist = np.mean(
        np.linalg.norm(pixel_coords - mean_pixel_coords, axis=1)
    )
    scale = mean_world_dist / mean_pixel_dist

    return translation, scale


def compute_rotation_matrix(world_coord):
    """Compute a roation matrix based on the supplied world coordinates matrix

    Args:
        world_coord (ndarry): world coordinate frame matrix

    Returns:
        ndarry: the rotation matrix
    """
    x, y = world_coord
    angle = np.arctan2(y, x)  # Calculate the angle based on the world
    # coordinates
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return rotation_matrix


def create_affine_matrix(translation, scale, rotation_matrix):
    """Create affing transformation matrix give rotation, translation, and scale

    Args:
        translation (ndarray): the translation matrix
        scale (ndarry): scaling
        rotation_matrix (ndarry): the rotation matrix

    Returns:
        ndarray: the affine transformation matrix
    """
    transformation_matrix = np.zeros((3, 3))
    transformation_matrix[:2, :2] = scale * rotation_matrix[:2, :2]
    transformation_matrix[:2, 2] = translation
    transformation_matrix[2, 2] = 1
    return transformation_matrix


def pixel_to_world_with_translation_scale(pixel_coords, translation, scale):
    """
    Convert pixel coordinates to world coordinates using translation and scale factors.

    Args:
        pixel_coords: A list of pixel coordinates as 2D arrays [[x1, y1], [x2, y2], ...].
        translation: A 2D array [tx, ty] representing the translation.
        scale: A 2D array [sx, sy] representing the scale factors.

    Returns:
        world_coords: A list of corresponding world coordinates as 2D arrays [[X1, Y1], [X2, Y2], ...].
    """
    pixel_coords = np.array(pixel_coords)
    world_coords = (pixel_coords + translation) * scale

    return world_coords
