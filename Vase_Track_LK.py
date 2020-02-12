import numpy as np
import cv2

#kernel = np.ones((5,5),np.uint8)
kernel = np.ones((3,3),np.float32)/25
cap = cv2.VideoCapture('projectV.avi')
r, init_frame = cap.read()
init_frame = cv2.dilate(init_frame,kernel,7)
# init_frame = cv2.medianBlur(init_frame,7)
# init_frame = cv2.bilateralFilter(init_frame,15,75,75)
# init_frame = cv2.filter2D(init_frame, -1, kernel)

init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
temp = init_frame

p_prev = np.array([[0.0]] * 6)
rect_prev = [125.0, 93.0, 170.0, 149.0]


def affineLKtracker(I, temp, rect, P):
    # to keep track of iterations
    ctr = 0

    # defining constants
    epsilon = 0.0548
    del_p = 0

    # getting the rectangle points
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    p1, p2, p3, p4, p5, p6 = P[0], P[1], P[2], P[3], P[4], P[5]
    W = (np.array([[(1 + p1), p3, p5], [p2, (1 + p4), p6]])).reshape(2, 3)

    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=9)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=9)

    while del_p is 0 or np.linalg.norm(del_p) >= epsilon :
    #while ctr < 4:

        ctr += 1

        for i in range(int(x1), int(x2) + 1, 15):
            for j in range(int(y1), int(y2) + 1, 15):
                p1, p2, p3, p4, p5, p6 = P[0], P[1], P[2], P[3], P[4], P[5]
                W = np.array([[(1 + p1), p3, p5], [p2, (1 + p4), p6]])
                W = W.reshape((2, 3))

                # step 1 : Warp I with W(x;p)
                X = np.array([[i], [j], [1]])
                Wxp = W @ X  # shape = (2, 1) (mapped x, mapped y)

                # step 2: template - (warped image of current iteration)
                T = temp[i, j]
                IW = I[int(Wxp[0]), int(Wxp[1])]
                error = np.array(T, dtype='int') - np.array(IW, dtype='int')
                error = error.reshape(1, 1)

                # step 3: compute gradient I
                I_grad = np.array([Ix[i, j], Iy[i, j]])

                # step 4: evaluate the jacobian dW/dP
                dwdp = np.array([[i, 0, j, 0, 1, 0], [0, i, 0, j, 0, 1]])

                # step 5: compute the steepest descent
                #
                sd = (np.transpose(I_grad) @ dwdp).reshape(1, 6)

                # step 6: calc hessian matrix
                hessian = np.array(np.sum((sd.T) @ sd))
                hessian = hessian.reshape(1, 1)
                hessian_inv = np.linalg.pinv(hessian)

                # step 7: multiply steepest descent with error
                steep_error = np.array(np.sum(np.transpose(sd) @ error)).reshape(1, 1)
                del_p = hessian_inv @ steep_error

                # step 9: update parameter matrix
                P += del_p

    # calculating the warped bounding box points

    Z1 = np.array([[x1], [y1], [1.0]], dtype='float')
    Z2 = np.array([[x2], [y2], [1.0]], dtype='float')
    W1 = W @ Z1
    W2 = W @ Z2
    x1_new, y1_new = W1[0], W1[1]
    x2_new, y2_new = W2[0], W2[1]

    rect_new = [x1_new, y1_new, x2_new, y2_new]

    return P, rect_new


i = 0
while cap.isOpened():

    r, cur_frame = cap.read()
    if r == True:
        i += 1
        OG = cur_frame
        cur_frame = cv2.dilate(cur_frame, kernel, 7)
        # curr_frame = cv2.medianBlur(cur_frame, 7)
        # cur_frame = cv2.bilateralFilter(cur_frame,9,75,75)
        # cur_frame = cv2.filter2D(cur_frame, -1, kernel)

        I = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        p_new, rect_new = affineLKtracker(I, temp, rect_prev, p_prev)
        p_prev = p_new
        rect_prev = rect_new
        print(rect_new)

        x1, y1, x2, y2 = rect_new[0], rect_new[1], rect_new[2], rect_new[3]
        img = cv2.rectangle(OG, (x1, y1), (x2, y2), (0, 0, 200), 2)
        #cv2.imwrite("T/Human{}.png".format(i) , img)
        cv2.imshow('', img)

        k = cv2.waitKey(30) & 0xFF
        if k is 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
