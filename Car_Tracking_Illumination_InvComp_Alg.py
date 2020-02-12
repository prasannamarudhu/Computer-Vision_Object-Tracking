import numpy as np
import cv2




def increase_brightness(img, value=30):
   img = cv2.GaussianBlur(img,(11,11),0)
   img = cv2.medianBlur(img,5)

   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   h, s, v = cv2.split(hsv)

   lim = 255 - value
   v[v > lim] = 255
   v[v <= lim] += value

   final_hsv = cv2.merge((h, s, v))
   img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
   return img


#Opening the video frame:
cap = cv2.VideoCapture("carvideotrack.mp4")

# The initial frame is captured and converted to gray scale
_, firstinit_frame = cap.read()

firstinit_frame=increase_brightness(firstinit_frame,30)

Gray_Init = cv2.cvtColor(firstinit_frame, cv2.COLOR_BGR2GRAY)

#The number of rowsand columns of gray scale image is obtained
r1, c1 = len(Gray_Init), len(Gray_Init[0])

#Initializing the rectangle co-ordinates on first frame
x,y=325,240
Trac_arrpts = np.array([(x,y)])
TrackPt = [p.ravel() for p in Trac_arrpts]
img = cv2.rectangle(firstinit_frame, (215, 280), (105, 50), (200, 0, 0), 3)

#This function executes the Lukas-Tracker Algorithm




def affineLKtracker(I, Curr_Fram, Pos1, Pos2, val, IX, IY):


    Temp1 = 0
    #Initializing the values
    Pos11 = np.matrix([[loop for loop in range(val)] for ran in range(val)])
    Pos21 = np.matrix([[ran] * val for ran in range(val)])

    #Checking for out of bounds values
    if ((Pos2 + val) > len(I)):
        return np.matrix([[-120], [-120]])
    if ((Pos1 + val) > len(I[0])):
        return np.matrix([[-120], [-120]])
    Step = -1

    #Template image
    TemplateImage = np.matrix([[I[i, j] for j in range(Pos1, Pos1 + val)] for i in range(Pos2, Pos2 + val)])

    #Initializing the parameter values
    param1 = 0.0
    param2 = 0.0
    param3 = 0.0
    param4 = 0.0
    param5 = 0.0
    param6 = 0.0

    #computing the gradient values
    #Gradient X
    IX = np.matrix([[IX[i, j] for j in range(Pos1, Pos1 + val)] for i in range(Pos2, Pos2 + val)])
    # Gradient Y
    IY = np.matrix([[IY[i, j] for j in range(Pos1, Pos1 + val)] for i in range(Pos2, Pos2 + val)])
    IP = [np.multiply(Pos11, IX), np.multiply(Pos11, IY), np.multiply(Pos21, IX),np.multiply(Pos21, IY), IX, IY]

    #Determining the Hessian matrix values
    HI = [[np.sum(np.multiply(IP[a], IP[b])) for a in range(6)] for b in range(6)]
    Inv_H = np.linalg.pinv(HI) #Inverse Hessian determination
    L = 0

    #Initializing the Warp matrix
    Warp_val = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Warp = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    while (Temp1 <= 10):

        position = [[Warp.dot(np.matrix([[Pos1 + i], [Pos2 + j], [1]], dtype='float')) for i in range(val)] for j in range(val)]


        if not (0 <= (position[0][0])[0, 0] < c1 and 0 <= (position[0][0])[1, 0] < r1 and 0 <= position[val - 1][0][0, 0] < c1 and 0 <= position[val - 1][0][1, 0] < r1 and 0 <= position[0][val - 1][0, 0] < c1 and 0 <=position[0][val - 1][1, 0] < r1 and 0 <= position[val - 1][val - 1][0, 0] < c1 and 0 <=position[val - 1][val - 1][1, 0] < r1):
            return np.matrix([[-120], [-120]])


        I = np.matrix([[Curr_Fram[int((position[i][j])[1, 0]), int((position[i][j])[0, 0])] for j in range(val)] for i in range(val)])

        #Calculating the error difference
        Diff = np.absolute(np.matrix(I, dtype='int') - np.matrix(TemplateImage, dtype='int'))
        s_Diff = np.matrix([[np.sum(np.multiply(g, Diff))] for g in IP]) #Calculating the steep_error difference

        #Delta_P calculation
        m_step = np.sum(np.absolute(s_Diff))
        p_del = Inv_H.dot(s_Diff)
        deltaP = Inverse_W(p_del)

        #Updating the parameter values with the delta_p value
        param1, param2, param3, param4, param5, param6 = param1 + deltaP[0, 0] + param1 * deltaP[0, 0] + param3 * deltaP[1, 0], param2 + deltaP[1, 0] + deltaP[0, 0] * param2 +param4 * deltaP[1, 0], param3 + deltaP[2, 0] + param1 * deltaP[2, 0] + param3 * deltaP[3, 0], param4 + deltaP[3, 0] + param2 * deltaP[2, 0] + param4 * deltaP[3, 0], param5 + \
                                 deltaP[4, 0] + param1 * deltaP[4, 0] + param3 * deltaP[5, 0], param6 + deltaP[5, 0] + param2 * deltaP[4, 0] + param4 * deltaP[5, 0]
        #obtaining the warped matrix
        Warp = np.matrix([[1 + param1, param3, param5], [param2, 1 + param4, param6]])


        if (Step == -1): Step = m_step
        elif (Step >= m_step):Step = m_step;L = 0;Warp_val = Warp
        else:L += 1
        if (L == 3):Warp = Warp_val;return Warp.dot(np.matrix([[Pos1], [Pos2], [1.0]]))
        if (np.sum(np.absolute(p_del)) < 0.0009): return Warp.dot(np.matrix([[Pos1], [Pos2], [1.0]]))

#This function generates the inverse warp matrix values
def Inverse_W(p):
    out_inv = np.matrix([[0.1]] * 6)
    val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]


    out_inv[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    out_inv[1, 0] = (-p[1, 0]) / val; out_inv[2, 0] = (-p[2, 0]) / val;out_inv[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    out_inv[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val;out_inv[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val

    return out_inv
index=20
#The main while condition
while (len(TrackPt) > 0):
    # The current frame is captured and converted to gray scale
    ret, frame = cap.read()
    OG = frame
    frame = increase_brightness(frame,20)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Computing gradient Gray_Init
    IX = cv2.Sobel(Gray_Init, cv2.CV_32F, 1, 0, ksize=5)
    IY = cv2.Sobel(Gray_Init, cv2.CV_32F, 0, 1, ksize=5)

    #Obtaining the new co-ordinate values
    CoOrd_new = [affineLKtracker(Gray_Init, frame_gray, int(x), int(y), 15, IX, IY) for x, y in TrackPt]
    pt_new = []

    #The Tracker rectangle is drawn on the frame
    for i in range(len(TrackPt)):
        a, b = TrackPt[i]
        c, d = int((CoOrd_new[i])[0]), int((CoOrd_new[i])[1])
        print("5", a,b,c,d)
        if (0 <= c < c1 and 0 <= d < r1):
            fin_img = cv2.line(frame, ((a-160),(b-130)), (a,(b-130)), (0, 255, 0), 2)
            fin_img = cv2.line(frame, (a,(b-130)), (a,b), (0, 255, 0), 2)
            fin_img = cv2.line(frame, (a,b),((a-160),b), (0, 255, 0), 2)
            fin_img = cv2.line(frame, ((a-160),b), ((a-160),(b-130)), (0, 255, 0), 2)
            pt_new.append((c,d))
            cv2.imshow('fin_img', fin_img)

    #img = cv2.add(frame,mask)
    #cv2.imshow('frame',fin_img)

    name = './Output/Car/Car_Normal_Frames/frame' + str(index) + '.jpg'
    print('Creating...' + name)
    cv2.imwrite(name, fin_img)
    index += 1

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    # Copying the frame_gray to Gray_Init
    Gray_Init = frame_gray.copy()
    TrackPt = pt_new[:]

cv2.destroyAllWindows()
cap.release()

