import cv2 
import numpy as np
#Hough algoritmasıyla video'da şerit tespitinin yapılması
cam = cv2.VideoCapture(r"opencv\video_ve_resimler\car8.mp4")
#Sapma değeri(100x100 piksel) kadarlık kısım yok sayılır.
#Kırpma işlemine o karenin sağ üst kÖşesinden başlanır.(1.koordinat sağ üst köşe)
sapma = 100     
kernel = np.ones((3,3), dtype=np.uint8)

#Frame'de kırpılacak alanın koordinatlarını belirleyen fonksiyon
def crop_matris(img):
    x, y = img.shape[:2]
    #Bu dizide x ve y lerin yerleri değişik
    #Koor. sistemi (x,y) şeklinde fakat y koordinatı (y-sapma) olması gerekirken (x-sapma) yazılmış.
    """kırpma işlemi örneğin yamuk şeklinde ise;
    1.köşe(yamuğun sol alt köşesi) = (sapma, x-sapma) -> sapma : x koordinatı, (x-sapma): y koordinatı
    2.köşe(yamuğun sol üst köşesi) = (y*3.2)/8 : x koordinatı , x*0.6 : y koordinatı
    3.köşe(yamuğun sağ üst köşesi) = (y*5)/8) :x koordinatı, x*0.6 : y koordinatı 
    4.köşe(yamuğun sağ alt köşesi) = y: x koordinatı , x-sapma :y koordinatı
    """
    value = np.array([
            [(sapma, x-sapma) , (int((y*3.2)/8),int(x*0.6)),
             (int((y*5)/8),int(x*0.6)) , (y,x-sapma)]],np.int32)
    return value

#Kırpılan kısmı orjinal görüntü, diğer kısımları siyah yapan fonksiyon  
def crop_image(img,matris):
    x, y = img.shape[:2]
    mask = np.zeros(shape = (x,y), dtype=np.uint8)
    mask = cv2.fillPoly(mask,matris,255)   #Matrisi maske üzerine çizdik.
    mask = cv2.bitwise_and(img, img, mask= mask)
    return mask  
    
def filt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.inRange(img, 120, 255)      # Değer 150-255 arasındaysa 1(beyaz), değilse 0 olur.
    img = cv2.erode(img, kernel)      #Resimdeki küçük noktaları sildik.
    img = cv2.dilate(img, kernel)     #Erode işleminden sonra resim küçüleceği için dilate ile tekrar büyüttük.
    img = cv2.medianBlur(img, 9)    
    img = cv2.Canny(img, 40, 100)
    return img

#Şeritin düzgün çizilmesi için bulunan değerlerin ortalamasını alan fonksiyon  
def line_mean(lines):
    left = []
    right = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y2 - y1)/(x2 - x1)     #eğim bulma işlemi
            
            if m < -0.2:
                #Eğim sola doğru yani sağ şerit(0.2 eşik değeri yatay çizgileri almamak için)
                right.append((x1,y1,x2,y2))
            elif m > 0.2:
                #Eğim sağa doğru yani sol şerit
                left.append(((x1,y1,x2,y2)))
        right_mean = np.mean(right, axis=0)    #right listesi içindeki değerleri topladık.
        left_mean = np.mean(left, axis=0)
    
    #Değerler None değilse
    if not isinstance(right_mean, type(np.nan)):
        if not isinstance(left_mean, type(np.nan)):
            return right_mean, left_mean
        else:
            return right_mean, None
    else:
        if not isinstance(left_mean, type(np.nan)):
            return None, left_mean
        else:
            return None, None

def draw_line(img, line):
    line = np.int32(np.around(line))  #Değeri tam sayıya yuvarladık.
    x1, y1, x2, y2 = line
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 10)
    return img

#Orjinal görüntüde kırptığımız alanı çokgen içinde gösteren fonksiyon
def draw_polylines(img, matris):
    x, y = img.shape[:2]
    dst = np.array([[matris[0][1,0],matris[0][1,1]],
                    [matris[0][0,0],matris[0][0,1]],
                    [matris[0][3,0],matris[0][3,1]],
                    [matris[0][2,0],matris[0][2,1]]],np.int32)
    cv2.polylines(img, [dst], True, (0,255,255), 2)
    return img

while cam.isOpened():
    ret, image = cam.read()
    if not ret:
        break
    
    img_org = image.copy()
    
    matris = crop_matris(image)
    img = crop_image(image, matris)   
    img = filt(img)
    
    #Çizgi tespiti, maxLineGap =  çizgi olarak algılanacak pikseller arası boşluk değeri
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 20, minLineLength=5, maxLineGap=100)
    
    image = draw_polylines(img_org, matris)
    
    if lines is not None:
        right_line, left_line = line_mean(lines)
        if right_line is not None:
            image = draw_line(image, right_line)
        if left_line is not None:
            image = draw_line(image, left_line)
            
    
    cv2.imshow("image",image)
    cv2.imshow("img",img)
    #33 ms = 30 fps (1 sn /30 frame) ,video 30 fps görüntülenecekse 33 ,60 fps olcaksa 16 yazılır.
    key = cv2.waitKey(16) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()