import math
import numpy as np
import cv2
import os


# 相関を求める半径(px)
template_radius = 35

# すこし大きめに見積もったBB弾の半径(px)
bb_radius = 50

# BB弾検出の閾値(DN)
min_gray = 45
max_gray = 255

def crop(img, pt, size):
    """
    画像の一部を矩形で切り取る
    
    Parameters
    ----------
    img : mat
        切り出し元の画像
    pt : [number,number]
        切り取り中心[col,row]
    size : [number,number]
        切り取りサイズ[width, height]

    Returns
    -------
    切り出した画像
    """
    left = int(pt[0] - size[0] / 2)
    right = int(pt[0] + size[0] / 2)
    top = int(pt[1] - size[1] / 2)
    bottom = int(pt[1] + size[1] / 2)
    d = len(img.shape)
    if d <= 2:
        return img[top:bottom, left:right]
    else:
        return img[top:bottom, left:right, :]

def rot(img, rot_in_deg):
    """
    画像を画像中央を中心に回転させる。回転により生じる背景は白で塗りつぶし。
    
    Parameters
    ----------
    img : mat
        元画像
    rot_in_deg : float
        回転角度(deg)
    Returns
    -------
    回転した画像
    """
    w ,h = 0, 0
    if(len(img.shape)):
        w, h = img.shape
        bg = 255
    else:
        w, h, _ = img.shape
        bg = (255, 255, 255)
    center = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D(center, rot_in_deg, 1)
    return cv2.warpAffine(img, mat, (w,h), borderValue=bg)

def inner_circle(src, pt, size = bb_radius):
    """
    領域の内接円中心を求める

    Parameters
    ----------
    src : mat
        元画像
    pt : [number,number]
        抽出する円のおおよその位置[col,row]
    size : int
        抽出する円を内包する直径(px)

    Returns
    -------
    内接円中心の座標[col,row]
    """
    c = crop(src, pt, (size * 2, size * 2))
    c = cv2.bitwise_not(c)
    contours, hierarchy=cv2.findContours(c,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.imshow('inner src', c)
    #cv2.waitKey(0)  
    for i in range(len(contours)):
        cnt=contours[i]
        moments=cv2.moments(contours[i])
        hu=cv2.HuMoments(moments)
        center = (int(moments['m10']/moments['m00'] + pt[0] - size),int(moments['m01']/moments['m00'] + pt[1] - size))
        #cv2.circle(src, center, 10, 200, 1)
        #cv2.imshow('center', src)
        #cv2.waitKey(0)
        return center
    return (-1,-1)


def estimate_rot(src, template, pt):
    """
    回転角の推定

    Parameters
    ----------
    src : mat
        推定対象の画像
    template : mat
        テンプレート画像
    pt : [number,number]
        推定対象の中心位置[col,row]
    Returns
    -------
    回転角 -180~+180(deg)
    """
    angle = -1
    max = -1.0e+308
    c = crop(src, pt, (template.shape[0], template.shape[1]))
    s = mask_circle(c, template_radius)
    cv2.imshow("match src", s)
    # ぐるぐる回しながら相関が最大となる角度を求める
    for i in range(-1800, 1800):
        t = rot(template, i / 10.0)
        cv2.imshow("template", t)
        #cv2.waitKey(1)
        match_result = cv2.matchTemplate(s, t, cv2.TM_CCOEFF)
        _, m, _, _ = cv2.minMaxLoc(match_result)
        # match_result = cv2.matchTemplate(s, t, cv2.TM_SQDIFF)
        # _, m, _, _ = cv2.minMaxLoc(match_result)
        # m = -m
        if(m > max):
            max = m
            angle = i / 10.0
    return angle

def mask_circle(src, radius):
    """
    円の外側を白で塗りつぶす。

    Parameters
    ----------
    src : mat
        塗りつぶし対象の画像

    radius : int
        円の半径(px)
    Returns
    -------
    塗りつぶした画像
    """
    mask = np.zeros(src.shape, np.uint8)
    center = (int(mask.shape[0]/2), int(mask.shape[1]/2))
    cv2.circle(mask, center, radius, 1, -1)
    ret = src * mask
    bg = np.ones(src.shape, np.uint8) * 255
    cv2.circle(bg, center, radius, 0, -1)
    return ret + bg

def process(filename):
    """
    写っているすべてのBB弾の角度を推定

    Parameters
    ----------
    filename : string
        画像ファイル名

    Returns
    -------
    結果をオーバーレイした画像
    """


    src = cv2.imread(filename, 0)
    # 右から撃った場合はコメントアウトして左から右へ時系列になるように反転
    # src = cv2.flip(src, 1)

    # 画像のコントラスト調整
    scaled = cv2.convertScaleAbs(src, alpha= 3, beta = 0)
    # ノイズ除去
    # この辺りは撮影データの品質依存…
    median = cv2.medianBlur(src, ksize=11)
    cv2.imshow('median', median)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (70, 70))
    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('opeing', opening)
    # 2値化してブロブ検出用の画像を作成
    ret, threshold = cv2.threshold(opening, min_gray, max_gray, cv2.THRESH_BINARY_INV)
    cv2.imshow('threshold', threshold)
    bsrc = threshold

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Area.
    
    params.filterByArea = True
    params.minArea = bb_radius * bb_radius * math.pi / 4
    params.maxArea = bb_radius * bb_radius * math.pi

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(bsrc) 

    # ブロブを赤丸で囲む
    blank = np.zeros((1, 1))  
    blobs = cv2.drawKeypoints(src, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

    # 列でソート
    kps = sorted(keypoints, key=lambda kp: kp.pt[0])

    # 左から数えてtemplate_index目のBB弾を基準とする
    template_index = 0

    # 基準としたBB弾を切り出してテンプレートとする
    c = crop(scaled, kps[template_index].pt, (kps[template_index].size + 10, kps[template_index].size + 10))

    # テンプレートの外周部をマスクする
    # BB弾の境界付近は相関が取りにくく結果が安定しないので多少削ってしまう
    template = mask_circle(c, 40)
    cv2.imshow("template", template);

    # 検出したBB弾すべてに角度推定を実行する
    result = []
    for k in kps:
        # ブロブの内接円中心を求める
        point = inner_circle(bsrc, k.pt)

        #　角度を推定
        angle = estimate_rot(scaled, template, point)

        # 推定角度がわかるように線分表示
        r = math.radians(angle)
        d = (math.sin(r) * 50,math.cos(r) * 50)
        pt1 = np.add(point, d).astype(np.int32)
        pt2 = np.subtract(point, d).astype(np.int32)
        cv2.line(blobs, pt1, pt2, (0,0,255), 1)

        #半回転以上する場合、計算が面倒なので補正
        #if(point[0] > 2500 and angle < 0):
        #    angle = angle + 180
        result.append((filename, point[0], point[1], angle))

    # 板にぶつかったコマを検出(板衝突検証用)
    min_index = result.index(min(result, key = lambda k: k[2]))

    # 結果の表示など
    for i, d in enumerate(result):
        print(*d, (i - min_index) * 0.1, sep = ',')
    cv2.imshow("blobs", blobs) 
    cv2.waitKey(0) 
    return blobs

res = process("sample.jpg")
cv2.imwrite("result.jpg", res)
#指定したフォルダの画像すべてに対して実行
# root = './bbrot'
# files = os.listdir(path=root)
# i = 1
# for f in files:
#     res = process(os.path.join(root, f))
#     cv2.imwrite(str(i) + ".jpg", res)
#     i += 1