import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Car_Inputvideo.mp4')
r, init_frame = cap.read()
init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)

temp = np.array([[init_frame[i, j] for j in range(99, 277)]for i in range(135, 330)])

p_prev = np.array([[0.0]] * 6)
rect_prev = np.array([(135, 99), (330, 277)])


def affineLKtracker(I, temp, rect, P):

	# to keep track of iterations
	ctr = 0

	# defining constants
	epsilon = 0.5
	del_p = None

	# getting the rectangle points
	x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[1][0], rect[1][1]


	p1, p2, p3, p4, p5, p6 = P[0], P[1], P[2], P[3], P[4], P[5]
	W = np.array([[(1 + p1), p3, p5], [p2, (1 + p4), p6]])
	W = W.reshape((2, 3))
	x = list()
	y = list()
	p = list()
	q = list()


	# x and y are array of indices for each new frame
	for i in range(int(x1), int(x2)):
		for j in range(int(y1), int(y2)):
			x.append(i)
			y.append(j)

	x = np.array(x)
	y = np.array(y)

	ones = np.ones(y.shape,)
	zeros = np.zeros(y.shape,)

	# p and q are array of indices in template image coordinate space
	for i in range(0, int(x2)-int(x1)):
		for j in range(0, int(y2)-int(y1)):
			p.append(i)
			q.append(j)

	p = np.array(p, dtype='int')
	q = np.array(q, dtype='int')

	T = np.array(temp[p, q])

	ttemp = np.array([[I[i, j] for j in range(int(y1), int(y2))]for i in range(int(x1), int(x2))])
	Ix = cv2.Sobel(ttemp, cv2.CV_64F, 1, 0, ksize=9)
	Iy = cv2.Sobel(ttemp, cv2.CV_64F, 0, 1, ksize=9)


	# this loop runs till del_p is minimised
	# while del_p is None or (np.linalg.norm(del_p) > epsilon):
	while ctr < 5:
		ctr+=1

		p1, p2, p3, p4, p5, p6 = P[0], P[1], P[2], P[3], P[4], P[5]
		W = np.array([[(1 + p1), p3, p5], [p2, (1 + p4), p6]])
		W = W.reshape((2, 3))


		X = np.array([x, y, ones])

		# step 1: warping I with W
		Wxp = W@X  # shape = (2, 1) (mapped x, mapped y)
		Wxp = Wxp.astype(int)
		IW = I[Wxp[0, :], Wxp[1, :]]

		# step 2: template - (warped image of current iteration)
		error = T - IW
		error = error.reshape(1, error.shape[0])

		# step 3: compute gradient I
		I_grad = np.array([Ix[p, q],Iy[p, q]])
		I_grad = I_grad.reshape(1, -1)


		# step 4: evaluate the jacobian dW/dP
		dwdp = (np.array([[x, zeros, y, zeros, ones, zeros], [zeros, x, zeros, y, zeros, ones]]))
		dwdp = dwdp.reshape(-1, dwdp.shape[1])
		# print("dwdp shape", dwdp.shape)

		# step 5: compute the steepest descent
		sd = I_grad@dwdp


		# step 6: calc hessian matrix
		product = np.dot(sd.T, sd)
		hessian = np.sum(product)
		hessian = hessian.reshape(1,1)
		hessian_inv = np.linalg.inv(hessian)

		# step 7: multiply steepest descent with error
		steep_error = np.array(np.sum(np.transpose(sd)@error))
		steep_error = steep_error.reshape(1, 1)

		# step 8: compute del_p
		del_p = hessian_inv@steep_error

		# step 9: update parameter matrix
		P += del_p


	# calculating the warped bounding box points
	Z1 = np.array([[x1], [y1], [1.0]], dtype='float')
	Z2 = np.array([[x2], [y2], [1.0]], dtype='float')
	W1 = W@Z1
	W2 = W@Z2
	x1_new, y1_new = W1[0], W1[1]
	x2_new, y2_new = W2[0], W2[1]

	rect_new = np.array([(x1_new, y1_new), (x2_new, y2_new)])
	return P, rect_new

frames = 0
while cap.isOpened():

	frames+=1

	r, cur_frame = cap.read()
	I = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

	p_new, rect_new = affineLKtracker(I, temp, rect_prev, p_prev)
	p_prev = p_new
	rect_prev = rect_new

	x1, y1, x2, y2 = rect_new[0][0], rect_new[0][1], rect_new[1][0], rect_new[1][1]
	# print(rect_new)

	img = cv2.rectangle(cur_frame, (x1, y1), (x2, y2), (200, 0, 0), 3)
	#cv2.imwrite("T/Car{}.png".format(frames) , img)
	cv2.imshow("tracker", img)
	print("evaluated frames :", frames)
	k = cv2.waitKey(30) & 0xFF
	if k is 27:
		break




cap.release()
cv2.destroyAllWindows()




