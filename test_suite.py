
from unittest import main, TestCase
import math
import numpy as np

import miscmath as mm
import contour_merge as cm

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

	# test mergeContours() from contour_merge.py
	def test_mergeContours (self) :
		contour1 = np.array([[[0, 0]], [[75, 3]], [[150, 0]], [[149, 303]]])
		contour2 = np.array([[[0, 300]], [[2, 160]], [[-2, 298]]])
		merged_contours = cm.mergeContours(contour1, contour2)
		expected_merged_contours = np.array([[[0, 0]], [[75, 3]], [[150, 0]], [[149, 303]], [[0, 300]], [[2, 160]], [[-2, 298]]])

		self.assertTrue(np.array_equal(merged_contours, expected_merged_contours))

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


main()