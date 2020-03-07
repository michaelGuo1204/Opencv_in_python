import cv2
import numpy as np
def to_Gray(input):
    row,col,channel=input.shape                                                       #获取图像信息
    deal_a_gary = np.zeros((row,col),dtype='uint8')#生成空白图像
    for r in range(row):
        for c in range(col):
            deal_a_gary[r,c]=input[r,c,0]*0.299+input[r,c,1]*0.587+input[r,c,2]*0.114#计算灰度值

    return deal_a_gary#图像灰度化
def find_contours(input):
    input_g=to_Gray(input)
    input_G=Gussian(input_g)
    input_S=sobel_gradint(input_G)
    output= link(input_S)
    return output
def Gussian(input):
    gSigma=1.4     #对高斯函数中的sigma赋值
    gwindowsize=5     #高斯核为3*3
    gcenter=int(gwindowsize/2)
    ggraph_width,ggraph_height=input.shape
    gMat=np.zeros(( ggraph_width,ggraph_height,),dtype='uint8')
    g_Kernel=np.zeros((gwindowsize*gwindowsize,1))   #生成二维滤波核
    g_sum=0                                                                 #准备均一化
    for i in range(gwindowsize):
        for j in range(gwindowsize):
            distance_x=i-gcenter;distance_y=j-gcenter
            g_Kernel[j*gwindowsize+i]=np.exp(-0.5*(distance_x*distance_x+distance_y*distance_y)/(gSigma*gSigma))/(2.0*3.1415926)*gSigma*gSigma#为高斯矩阵赋值
            g_sum=g_sum+g_Kernel[j*gwindowsize+i]
    for i in range(gwindowsize):
        for j in range(gwindowsize):
            g_Kernel[j*gwindowsize+i]=g_Kernel[j*gwindowsize+i]/g_sum#均一化
            print(g_Kernel[j*gwindowsize+i])                                               #debug需要
    for s in range(ggraph_width):
        for t in range(ggraph_height):
            dFilter=0
            dSum=0
            for x in range(-gcenter,gcenter):
                for y in range(-gcenter,gcenter):
                    if(x+s>=0 and x+s<ggraph_width and y+t>=0  and y+t<ggraph_height):
                        currenrvalue=input[(x+s),(y+t)]
                        dFilter=dFilter+currenrvalue*g_Kernel[x+gcenter+(y+gcenter)*gcenter]
                        dSum=dSum+g_Kernel[x+gcenter+(y+gcenter)*gcenter]
            gMat[s,t]=dFilter/dSum
    return gMat#高斯滤波
def sobel_gradint(input):   
    kernel=np.array([[1.0,2.0,1.0],[2.0,4.0,2.0],[1.0,2.0,1.0]])/16
    kernelx=np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8
    kernely=np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])/8   #4个sobel核
    input_s=cv2.filter2D(input,-1,kernel)
    my_sobelx=cv2.filter2D(input,-1,kernelx)
    my_sobely=cv2.filter2D(input,-1,kernely)
    input_s=input_s.astype(np.uint8)
    my_sobelx=my_sobelx.astype(np.uint8)
    W1, H1 = input.shape
    dx = np.zeros([W1-1, H1-1])
    dy = np.zeros([W1-1, H1-1])
    d = np.zeros([W1-1, H1-1])
    for i in range(W1-1):
          for j in range(H1-1):   
                dx[i,j] = int(input[i, j+1]) - int(input[i, j])
                dy[i,j] = int(input[i+1, j]) - int(input[i, j])        
                d[i, j] = np.sqrt(np.square(dx[i,j]) + np.square(dy[i,j]))   # 图像梯度幅值作为图像强度值
    cv2.imshow("a",d);cv2.waitKey(10)
    W2, H2 = d.shape
    Mat_deal = np.copy(d)
    Mat_deal[0,:] = Mat_deal[W2-1,:] = Mat_deal[:,0] = Mat_deal[:, H2-1] = 0
    for i in range(1, W2-1):
         for j in range(1, H2-1):
              if d[i, j] == 0:
                  Mat_deal[i, j] = 0
              else:
                  gradX = dx[i, j]
                  gradY = dy[i, j]
                  gradTemp = d[i, j]
            
            # 如果Y方向幅度值较大
                  if np.abs(gradY) > np.abs(gradX):
                      weight = np.abs(gradX) / np.abs(gradY)
                      grad2 = d[i-1, j]
                      grad4 = d[i+1, j]
                # 如果x,y方向梯度符号相同
                      if gradX * gradY > 0:
                          grad1 = d[i-1, j-1]
                          grad3 = d[i+1, j+1]
                # 如果x,y方向梯度符号相反
                      else:
                          grad1 = d[i-1, j+1]
                          grad3 = d[i+1, j-1]
                    
            # 如果X方向幅度值较大
                  else:
                        weight = np.abs(gradY) / np.abs(gradX)
                        grad2 = d[i, j-1]
                        grad4 = d[i, j+1]
                  # 如果x,y方向梯度符号相同
                        if gradX * gradY > 0:
                            grad1 = d[i+1, j-1]
                            grad3 = d[i-1, j+1]
                # 如果x,y方向梯度符号相反
                        else:
                            grad1 = d[i-1, j-1]
                            grad3 = d[i+1, j+1]
        
                  gradTemp1 = weight * grad1 + (1-weight) * grad2
                  gradTemp2 = weight * grad3 + (1-weight) * grad4
                  if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                      Mat_deal[i, j] = gradTemp
                  else:
                      Mat_deal[i, j] = 0
    return Mat_deal#sobel 卷积
def link(Mat_deal):
    W3, H3 = Mat_deal.shape
    DT = np.zeros([W3, H3])     
    TL = 0.1* np.max(Mat_deal)
    TH = 0.2 * np.max(Mat_deal)
    for i in range(1, W3-1):
          for j in range(1, H3-1):
               if (Mat_deal[i, j] < TL):
                   DT[i, j] = 0
               elif (Mat_deal[i, j] > TH):
                   DT[i, j] = 1
               elif ((Mat_deal[i-1, j-1:j+1] < TH).any() or (Mat_deal[i+1, j-1:j+1]).any() 
                  or (Mat_deal[i, [j-1, j+1]] < TH).any()):
                DT[i, j] = 1
    cv2.imshow("result",DT)
    return DT#链接轮廓

input=cv2.imread(r"Lena.jpg")
cv2.imshow("cc",input)
cv2.waitKey(10)
deal_a=input       #deal_a will be manully treated
deal_b=input       #deal_b will be treated by cv2
deal_b_gray=cv2.cvtColor(deal_b,cv2.COLOR_RGB2GRAY)             #treat deal_b
cv2.imshow("cv2",deal_b_gray);cv2.waitKey(10)
to_Gray(deal_a)   #treat deal_a
deal_b_contour=cv2.Canny(deal_b,100,200)# treat deal_b
cv2.imshow("canny",deal_b_contour);cv2.waitKey(10)
a_contour=find_contours(deal_a) #treat deal_a
cv2.imshow("my canny",a_contour);cv2.waitKey(10)
####################################################################
######################视频处理########################################
####################################################################
video = cv2.VideoCapture("test.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc =  cv2.VideoWriter_fourcc(*'MP4V')
videoWriter = cv2.VideoWriter('./trans.mp4', fourcc, fps, size)  
success, frame = video.read()  
index = 1
while success :  
    frame=cv2.Canny(frame,20,200)
    cv2.imshow("new video", frame)
    cv2.waitKey(40)
    videoWriter.write(frame)
    success, frame = video.read()
    index =index+ 1
video.release()
##################################################################################
################################调用摄像头#######################################
video = cv2.VideoCapture(0)
success, frame = video.read()  
while success :  
    frame=cv2.Canny(frame,100,200)
    cv2.imshow("new video", frame)
    if cv2.waitKey(10)==27:
        break
    success, frame = video.read()  
video.release()




    



