
from unittest import main, TestCase
import math
import numpy as np

import miscmath as mm
import contour_merge as cm
import vectormath as vm
import track_planar as tp


class TestMiscMath (TestCase) :

	# test dist() from miscmath.py
	def test_dist_1 (self) :
		self.assertEqual(mm.dist(6, 6, 8, 8), math.sqrt(8))

	def test_dist_2 (self) :
		self.assertEqual(mm.dist(1, 1, 2, 2), math.sqrt(2))

	def test_dist_3 (self) :
		self.assertEqual(mm.dist(2, 2, 4, 5), math.sqrt(13))


	# test avg() from miscmath.py
	def test_avg_1 (self) :
		data_points = [0, 5, 10]
		self.assertEqual(mm.avg(i for i in data_points), 5)

	def test_avg_2 (self) :
		data_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		self.assertEqual(mm.avg(i for i in data_points), 4.5)

	def test_avg_3 (self) :
		data_points = [-10, 10, 1, 100,]
		self.assertEqual(mm.avg(i for i in data_points), 25.25)


class TestContourMerge (TestCase) :

	# test contourRectangularity() from contour_merge.py
	def test_contorsRectangularity_1 (self) :
		contour = np.array([[[0, 0]], [[150, 0]], [[150, 300]], [[0, 300]]])
		self.assertEqual(cm.contourRectangularity(contour), 0)

	def test_contorsRectangularity_2 (self) :
		contour = np.array([[[0, 0]], [[170, 0]], [[200, 200]], [[0, 180]]])
		self.assertTrue(cm.contourRectangularity(contour) < 2)

	def test_contorsRectangularity_3 (self) :
		contour = np.array([[[0, 0]], [[500, 0]], [[500, 1000]], [[0, 1000]]])
		self.assertEqual(cm.contourRectangularity(contour), 0)


	# test appendContours() from contour_merge.py
	def test_appendContours_1 (self) :
		contour1 = np.array([[[0, 0]], [[1, 1]], [[2, 2]]])
		contour2 = np.array([[[3, 3]], [[4, 4]], [[5, 5]]])
		appended_contours = cm.appendContours(contour1, contour2)
		expected_result = np.array([[[0, 0]], [[1, 1]], [[2, 2]], [[3, 3]], [[4, 4]], [[5, 5]]])

		self.assertTrue(np.array_equal(appended_contours, expected_result))

	def test_appendContours_2 (self) :
		contour1 = np.array([[[0, 0]], [[1, 1]], [[2, 2]]])
		contour2 = np.array([[[3, 3]], [[4, 4]], [[5, 5]]])
		appended_contours = cm.appendContours(contour1, contour2, True)
		expected_result = np.array([[[2, 2]], [[1, 1]], [[0, 0]], [[3, 3]], [[4, 4]], [[5, 5]]])

		self.assertTrue(np.array_equal(appended_contours, expected_result))

	def test_appendContours_3 (self) :
		contour1 = np.array([[[0, 0]], [[1, 1]], [[2, 2]]])
		contour2 = np.array([[[3, 3]], [[4, 4]], [[5, 5]]])
		appended_contours = cm.appendContours(contour1, contour2, False, True)
		expected_result = np.array([[[0, 0]], [[1, 1]], [[2, 2]], [[5, 5]], [[4, 4]], [[3, 3]]])

		self.assertTrue(np.array_equal(appended_contours, expected_result))

	def test_appendContours_4 (self) :
		contour1 = np.array([[[0, 0]], [[1, 1]], [[2, 2]]])
		contour2 = np.array([[[3, 3]], [[4, 4]], [[5, 5]]])
		appended_contours = cm.appendContours(contour1, contour2, True, True)
		expected_result = np.array([[[2, 2]], [[1, 1]], [[0, 0]], [[5, 5]], [[4, 4]], [[3, 3]]])

		self.assertTrue(np.array_equal(appended_contours, expected_result))


	# test mergeContours() from contour_merge.py
	def test_mergeContours (self) :
		contour1 = np.array([[[0, 0]], [[75, 3]], [[150, 0]], [[149, 303]]])
		contour2 = np.array([[[0, 300]], [[2, 160]], [[-2, 298]]])
		merged_contours = cm.mergeContours(contour1, contour2)
		expected_result = np.array([[[0, 0]], [[75, 3]], [[150, 0]], [[149, 303]], [[0, 300]], [[2, 160]], [[-2, 298]]])

		self.assertTrue(np.array_equal(merged_contours, expected_result))


	# test contoursMergeable() from contour_merge.py
	def test_contoursMergeable_1 (self) :
		contour1 = np.array([[[0, 0]], [[150, 0]], [[150, 300]]])
		contour2 = np.array([[[-2, 300]], [[5, -50]], [[0, 300]]])
		merged_contours = cm.mergeContours(contour1, contour2)

		self.assertTrue(cm.contoursMergeable(contour1, contour2, merged_contours))

	def test_contoursMergeable_2 (self) :
		contour1 = np.array([[[0, 0]], [[75, 3]], [[150, 0]], [[149, 303]]])
		contour2 = np.array([[[0, 300]], [[2, 160]], [[-2, 298]]])
		merged_contours = cm.mergeContours(contour1, contour2)

		self.assertTrue(cm.contoursMergeable(contour1, contour2, merged_contours))


	# test collapseContours() from contour_merge.py
	def test_collapseContours (self) :
		contour1 = np.array([[[0, 0]], [[75, 3]], [[150, 0]], [[149, 303]]])
		contour2 = np.array([[[0, 300]], [[2, 160]], [[-2, 298]]])
		contour_list = [contour1, contour2]
		collapsed_contours = cm.collapseContours(contour_list)
		expected_result = np.array([[[[0, 0]], [[75, 3]], [[150, 0]], [[149, 303]], [[0, 300]], [[2, 160]], [[-2, 298]]]])

		self.assertTrue(np.array_equal(collapsed_contours, expected_result))


class TestVectorMath (TestCase) :

	# test unit_vector() from vectormath.py
	def test_unit_vector (self) :
		unit_vector = vm.unit_vector([3, 3, 3])
		expected_result = [3.0/math.sqrt(27), 3.0/math.sqrt(27), 3.0/math.sqrt(27)]
		self.assertTrue(np.array_equal(unit_vector, expected_result))

	# test angle_between() from vectormath.py
	def test_angle_between (self) :
		vector1 = [0, 1, 0]
		vector2 = [1, 1, 0]
		angle_between = vm.angle_between(vector1, vector2)
		expected_result = math.pi/4.0

		self.assertTrue(angle_between - expected_result < .001)

	# test test_colinear() from vectormath.py
	def test_test_colinear (self) :
		p1 = np.array([1, 2])
		p2 = np.array([1, 5])
		p3 = np.array([8, 32])

		boolean, middle_point = vm.test_colinear((p1, p2, p3))

		self.assertTrue(np.array_equal(middle_point, p2))

	# test approx_quadrilateral() from vectormath.py
	# def test_approx_quadrilateral (self) :
	# 	p1 = np.array([0, 0])
	# 	p2 = np.array([10, 0])
	# 	p3 = np.array([10, 10])
	# 	corners = (p1, p2, p3)
	# 	result = vm.approx_quadrilateral(corners, 20)
	# 	expected_corners = (p1, p2, p3, np.array([0, 10]))

	# 	self.assertTrue(np.array_equal(result, expected_corners))


	class TestTrackPlanar (TestCase) :

		# test null_callback() from track_planar.py
		def test_null_callback (self) :
			result = tp.null_callback(1)
			self.assertEqual(result, None)


main()