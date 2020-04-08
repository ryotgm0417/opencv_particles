import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys

# --------
# Parameters / 設定のための変数

# Area of paper (i.e. field of view / area of interest)
PAPER_AREA = 52.0 * 76.0

# Histogram intervals
AREA_STEP = 0.02    # Interval for the "Area" histogram
DIAMETER_STEP = 0.05    # Interval for the "Equivalent" histogram

# Parameter for cv2.AdaptiveThreshold
# Lower value = Higher sensitivity
THRESH_LEVEL = 13

# Minimum particle size
# Ignores results with an equivalent diameter smaller than this value
MINIMUM_DIAMETER = 0.050

# Option to select regions to ignore
SELECT_REGION = False

# 丸っぽさ（compactness）を見るかどうか
# True なら、丸っぽい点のみを数える。False にすると形を気にしなくなる
# 基本的には、変更しなくて良いはず

# Whether to check the compactness of results
# (Compactness is an indicator of how "circular" a geometrical shape is)
# if True, ignores results with a compactness smaller than a certain value
ENABLE_COMPACTNESS = True

# HSV filter parameters
low_H = 0
low_S = 70
low_V = 100
high_H = 100
high_S = 255
high_V = 255



# --------
# Program Code
# --------
args = sys.argv
directory = "images/"
# np.seterr('raise')

# Import image / 画像取り込み
img_raw = cv2.imread(directory + args[1], 1)
height = img_raw.shape[0]
width = img_raw.shape[1]

# --------
# Select regions to ignore using mouse / ゴミの領域などをマウスで除外
img_rect = copy.deepcopy(img_raw)
on_click = []
off_click = []

def on_mouse(event, x, y, flags, param):
    global on_click, off_click
    if event == cv2.EVENT_LBUTTONDOWN:
        on_click.append((x,y))

    elif event == cv2.EVENT_LBUTTONUP:
        off_click.append((x,y))
        cv2.rectangle(img_rect, on_click[-1], off_click[-1], (0,255,0), 2)

if SELECT_REGION:
    cv2.namedWindow("Select regions to ignore", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select regions to ignore", on_mouse)

    while True:
        cv2.imshow("Select regions to ignore", img_rect)
        key = cv2.waitKey(1)

        # 'r' key = RESET
        if key == ord("r"):
            on_click = []
            off_click = []
            img_rect = copy.deepcopy(img_raw)
        
        # 'q' key = QUIT
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


# --------
# Extract paper region / 紙の領域を抽出
img_hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
th_hsv = cv2.inRange(img_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

k1 = np.ones((5,5), np.uint8)
paper = cv2.morphologyEx(th_hsv, cv2.MORPH_OPEN, k1)

paper_ct, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.fillPoly(paper, pts=paper_ct, color=255)

paper_area = np.sum(paper/255)

# Slightly reduce the paper region / 紙の領域を縮小
k3 = np.ones((40,40), np.uint8)
paper = cv2.erode(paper, k3, iterations = 1)

# print(f'Area of Whole Image [px^2]: {height*width}')
# print(f'Area of Paper [px^2]: {paper_area}')

# Remove region selected with mouse / マウスで選択した領域は紙から外す
if len(on_click) > 0:
    for i in range(len(on_click)):
        paper[on_click[i][1]:off_click[i][1], on_click[i][0]:off_click[i][0]] = 0

# Show paper region / 紙の領域を元画像に重ねて表示
paper_red = cv2.cvtColor(paper, cv2.COLOR_GRAY2RGB)
paper_red[np.where((paper_red == [255,255,255]).all(axis=2))] = [0,0,255]
paper_region = cv2.addWeighted(img_raw,0.5,paper_red,0.5,0)

# --------
# Grayscale image of original / 元画像をグレースケール化
img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
mean_intensity = img_gray[np.where(paper == 255)].mean()

# Create histogram of grayscale image / グレースケール画像のヒストグラムを作る
img_hist = copy.deepcopy(img_gray)
img_hist[np.where(paper != 255)] = 0

# To binary image / 画像の二値化
img_adpt = copy.deepcopy(img_gray)
img_adpt[np.where(paper != 255)] = mean_intensity
th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, THRESH_LEVEL)
# ret, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Fill holes in particles / 粒を単色に埋める
kernel = np.ones((3,3), np.uint8)
th_fill = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

# Ignore area outside paper / 紙の領域より外の部分を除外
th_fill = cv2.bitwise_and(th_fill, paper)

# Find and draw contours / 粒の輪郭を検出・描画
th_copy = copy.deepcopy(th_fill)
contours, _ = cv2.findContours(th_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate area and equivalent diameter of each particle
# Area is [mm^2], Diameter is [mm]
ct_no = len(contours)
Areas = np.zeros(ct_no)
Eq_diameters = np.zeros(ct_no)

num_particles = 0
ignored_size = 0
ignored_compactness = 0

for i in range(ct_no):
    cnt = contours[i]
    raw_a = cv2.contourArea(cnt)
    a = raw_a / paper_area * PAPER_AREA
    eqd = np.sqrt(4*a/np.pi)
    
    M = cv2.moments(cnt)
    I = M['mu20'] + M['mu02']
    compactness = raw_a*raw_a/(2*np.pi*I + 0.1)
    if not ENABLE_COMPACTNESS: compactness = 1.

    if eqd >= MINIMUM_DIAMETER and compactness > 0.5:
        Areas[i] = a
        Eq_diameters[i] = eqd
        num_particles += 1
        img_raw = cv2.drawContours(img_raw, [cnt], -1, (0,0,255), 1)
    else:
        Areas[i] = -1.
        Eq_diameters[i] = -1.
        if(eqd < MINIMUM_DIAMETER):
            ignored_size += 1
        else:
            ignored_compactness += 1
    
        img_raw = cv2.drawContours(img_raw, [cnt], -1, (0,255,0), 1)

area_sum = Areas[Areas > 0].sum()

area_bins = np.arange(0., Areas.max() + AREA_STEP, AREA_STEP)
diam_bins = np.arange(0., Eq_diameters.max() + DIAMETER_STEP, DIAMETER_STEP)

area_hist, area_bedges = np.histogram(Areas, bins=area_bins)
diam_hist, diam_bedges = np.histogram(Eq_diameters, bins=diam_bins)

# Show Results / 結果の表示
print('==============================')
print(f'Total Number of Particles: {num_particles}')
print(f'Area Sum [mm^2]: {area_sum}')
print(f'(For reference) Ignored Particles - Too Small: {ignored_size}')
print(f'(For reference) Ignored Particles - Not Round: {ignored_compactness}')
print('==============================')
print(' Area of Particles')
print('------------------------------')
for i in range(len(area_hist)):
    print(f'{area_bins[i]:.3f} <= A [mm^2] < {area_bins[i+1]:.3f}  ||  {area_hist[i]}')

print('==============================')
print(' Equivalent Diameter of Particles')
print('------------------------------')
for i in range(len(diam_hist)):
    print(f'{diam_bins[i]:.3f} <= d [mm] < {diam_bins[i+1]:.3f}  ||  {diam_hist[i]}')

print('==============================')


# Show images / 解析画像を表示

# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# cv2.imshow("Original",img_raw)
cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale",img_adpt)
cv2.namedWindow("Target Region", cv2.WINDOW_NORMAL)
cv2.imshow("Target Region", paper_region)
cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
cv2.imshow("Binary Image", th_fill)
cv2.namedWindow("Found Particles (Contour)", cv2.WINDOW_NORMAL)
cv2.imshow("Found Particles (Contour)", img_raw)

cv2.waitKey(0)


# Draw Histogram / グラフを表示

area_graph_x = []
for i in range(len(area_bins)-1):
    area_graph_x.append((area_bins[i]+area_bins[i+1])/2.)

diam_graph_x = []
for i in range(len(diam_bins)-1):
    diam_graph_x.append((diam_bins[i]+diam_bins[i+1])/2.)


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))
ax1.set_title('Histogram of Areas')
ax1.bar(area_graph_x, area_hist, AREA_STEP*0.8)
ax2.set_title('Histogram of Equivalent Diameters')
ax2.bar(diam_graph_x, diam_hist, DIAMETER_STEP*0.8)

plt.show()