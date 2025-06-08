# me  –  Script TOP DAT 這是 TouchDesigner 的 Python 
# 這個檔案是用來處理 Hokuyo 光達的資料，並將其轉換為 OpenCV 圖像格式。
import math, cv2, numpy as np
from typing import NamedTuple,List, Tuple
import timeit
import socket
import base64
import hashlib
import struct
import json

# ========================架構=========================
# 1. 參數
# 2. 初始化
# 9999. 主渲染脈衝

# ========================參數=========================
human_circle_fill = True
# 在劇場中(生產環境)設為假，用虛擬資料(開發環境)設為真
bool_use_virtual_data = True
# 是否進行效能量測
bool_performance_measure = False
# 演算法選擇
algorithmSelect = 3

# ========================初始化=========================
def onSetupParameters(scriptOp):
    # 無自訂參數
    return
    
#WS**************************************************************
def normalize_points(points: List[Tuple[int, int]], x_max: int, y_max: int) -> List[List[float]]:
    if x_max == 0 or y_max == 0:
        raise ValueError("x_max and y_max must be non-zero")

    normalized = []
    for x, y in points:
        norm_x = x / x_max
        norm_y = y / y_max
        normalized.append([1.0-norm_x, 1.0-norm_y])
    return normalized
def create_ws_key():
    key = base64.b64encode(b"handcrafted_key_1234")
    return key.decode()

def build_frame(data):
    # Simple frame: FIN=1, opcode=0x1 (text), no fragmentation
    payload = data.encode()
    frame = bytearray()
    frame.append(0x81)  # FIN=1, text frame (0x1)
    
    length = len(payload)
    if length < 126:
        frame.append(0x80 | length)
    elif length < (1 << 16):
        frame.append(0x80 | 126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack(">Q", length))

    # Masking key (required for client-to-server)
    masking_key = b'\x00\x00\x00\x00'  # For demo; normally random
    frame.extend(masking_key)

    # Masked payload
    frame.extend(b ^ masking_key[i % 4] for i, b in enumerate(payload))
    return frame

def send_websocket_message(host, port, path, message):
    key = create_ws_key()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    # WebSocket HTTP Upgrade handshake
    handshake = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    s.send(handshake.encode())
    response = s.recv(1024)
    if b"101 Switching Protocols" not in response:
        print("Handshake failed")
        return

    # Send frame
    frame = build_frame(message)
    s.send(frame)
    print("Message sent")
    s.close()
#WS**************************************************************

# ───────────────────  HELPERS  ──────────────────────────────────

def flip_image_both_axes(img: np.ndarray) -> np.ndarray:
    """
    對一張 BGR 圖像進行水平與垂直翻轉（不含 alpha 通道）

    Parameters:
        img (np.ndarray): 3 通道 BGR 格式的 OpenCV 圖像

    Returns:
        np.ndarray: 翻轉後的圖像
    """
    if img is None:
        raise ValueError("輸入圖像為 None")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("輸入圖像需為 BGR 格式（三通道）")

    # flipCode = -1 表示水平 + 垂直翻轉
    return cv2.flip(img, -1)
def flip_coordinates_batch(points: List[Tuple[int, int]], w: int, h: int) -> List[Tuple[int, int]]:
    """
    對多個座標點進行水平與垂直翻轉。
    
    Args:
        points (List[Tuple[int, int]]): 原始座標點清單
        w (int): canvas 寬度
        h (int): canvas 高度

    Returns:
        List[Tuple[int, int]]: 翻轉後的座標清單
    """
    return [(w - x, h - y) for x, y in points]
def process_image_white_BG_2_black_BG(img: np.ndarray) -> np.ndarray:
    """
    處理不含 alpha 的 BGR 圖像：
    - 將非黑非白像素設為白色
    - 黑色變白色，白色變黑色

    Parameters:
        img (np.ndarray): 3 通道 BGR 圖像

    Returns:
        np.ndarray: 處理後的圖像
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("輸入圖像需為 BGR 格式（三通道）")

    bgr = img.copy()

    # 建立黑與白的遮罩
    black_mask = np.all(bgr == [0, 0, 0], axis=-1)
    white_mask = np.all(bgr == [255, 255, 255], axis=-1)
    other_mask = ~(black_mask | white_mask)

    # 將非黑非白像素設為白色
    bgr[other_mask] = [255, 255, 255]

    # 更新遮罩
    black_mask = np.all(bgr == [0, 0, 0], axis=-1)
    white_mask = np.all(bgr == [255, 255, 255], axis=-1)

    # 黑白反轉
    bgr[black_mask] = [255, 255, 255]
    bgr[white_mask] = [0, 0, 0]

    return bgr
def scale_points_to_target(
    ptls: List[Tuple[int, int]],
    canvas_width: int,
    canvas_height: int,
    target_w: int,
    target_h: int
) -> List[Tuple[int, int]]:
    """
    Scale a list of (x, y) points to fit within a new target width and height,
    relative to the original canvas size.
    
    Args:
        ptls: List of tuples representing (x, y) coordinates.
        canvas_width: Original canvas width.
        canvas_height: Original canvas height.
        target_w: Target canvas width.
        target_h: Target canvas height.
    
    Returns:
        A list of scaled (x, y) points.
    """
    if canvas_width == 0 or canvas_height == 0:
        raise ValueError("Original canvas width and height must be non-zero")

    # Calculate scale factors for width and height
    scale_x = target_w / canvas_width
    scale_y = target_h / canvas_height

    # Scale all points
    scaled_ptls = [(int(x * scale_x), int(y * scale_y)) for x, y in ptls]
    
    return scaled_ptls
class STRUCT_group_and_draw_circles(NamedTuple):
    img: np.ndarray
    r: int
    ptls: List[Tuple[int, int]]
    maskFull: np.ndarray
    target_h: int
    target_w: int
def filter_points(
    ptls: List[Tuple[int, int]],
    x_full: int,
    y_full: int,
    x_percent: float,
    y_percent: float
) -> List[Tuple[int, int]]:
    """
    Filters out points from ptls where:
    - x is less than x_full * x_percent or greater than x_full * (1 - x_percent)
    - y is less than y_full * y_percent or greater than y_full * (1 - y_percent)
    
    Returns a filtered list of (x, y) tuples.
    """
    x_min = x_full * x_percent/100.0
    x_max = x_full * (1 - x_percent/100.0)
    y_min = y_full * y_percent/100.0
    y_max = y_full * (1 - y_percent/100.0)

    return [
        (x, y) for x, y in ptls
        if x_min <= x <= x_max and y_min <= y <= y_max
    ]

# ===================end of helpers=========================

# =====================KD TREE==========================
class KDNode:
    def __init__(self, point, idx, axis):
        """
        point: (x, y) 二維座標
        idx: 這個節點對應到原始點列表裡的索引
        axis: 分割維度 (0 = x, 1 = y)
        """
        self.point = point    # tuple (x, y)
        self.idx = idx        # int, 原始陣列的索引
        self.axis = axis      # 0 或 1
        self.left = None
        self.right = None

def build_kdtree(points, depth=0):
    """
    points: list of (x, y, idx) 三元素 tuple。
            其中 idx 是這筆資料在 coords_list 裡的索引，用於回傳 cluster 時做標記。
    depth: 建樹深度，用來決定目前要用哪個維度切割 (axis = depth % 2)
    回傳: KDNode (子樹根節點)
    """
    if not points:
        return None

    axis = depth % 2
    # 依照 axis 維度排序，取中位數
    points.sort(key=lambda elem: elem[axis])
    mid = len(points) // 2

    x, y, idx = points[mid]
    node = KDNode(point=(x, y), idx=idx, axis=axis)
    # 左子樹、右子樹遞迴建
    node.left = build_kdtree(points[:mid], depth + 1)
    node.right = build_kdtree(points[mid + 1 :], depth + 1)
    return node

def kdtree_radius_search(node, target_pt, radius, results):
    """
    在 KD-Tree 裡搜尋所有與 target_pt 距離 <= radius 的點，把它們的 idx 加到 results 中。
    node: KDNode 節點
    target_pt: (x, y)
    radius: 半徑 (int 或 float)
    results: list，要把符合條件的 idx 推進去
    """
    if node is None:
        return

    x0, y0 = target_pt
    x1, y1 = node.point
    dx = x0 - x1
    dy = y0 - y1
    # 先檢查此節點本身是否在 radius 內
    if dx * dx + dy * dy <= radius * radius:
        results.append(node.idx)

    axis = node.axis
    # 判斷 target_pt 在切割平面哪一側
    diff = target_pt[axis] - node.point[axis]
    # 先往「可能包含自身」的一側子樹搜尋
    if diff <= 0:
        kdtree_radius_search(node.left, target_pt, radius, results)
    else:
        kdtree_radius_search(node.right, target_pt, radius, results)
    # 判斷對側子樹是不是也可能有點落在 radius 內
    if diff * diff <= radius * radius:
        if diff <= 0:
            kdtree_radius_search(node.right, target_pt, radius, results)
        else:
            kdtree_radius_search(node.left, target_pt, radius, results)
# =====================END OF KD TREE==========================

def group_and_draw_circles_KD(objGDC:STRUCT_group_and_draw_circles) -> List[Tuple[int, int]]:
    """
    1) 根據 x_pct, y_pct 做邊界 cropping，產生 mask_full
    2) 擷取 mask_full 裡所有非零像素座標 (x, y)
    3) 用純 Python KD-Tree 把這些座標做半徑 r 的分群 (clustering)
    4) 對每個 cluster 計算中心點 (mean_x, mean_y)，並以半徑 r 畫圓
    5) 再做翻轉與顏色反轉
    """
    # 先把所有前景座標轉成 (x, y) 列表，並同時記錄索引
    coords = objGDC.ptls
    points_for_tree = [(coords[i][0], coords[i][1], i) for i in range(len(coords))]

    # 1) 建 KD-Tree
    kdtree_root = build_kdtree(points_for_tree, depth=0)

    # 2) 用半徑 r 做群集 (cluster)，BFS + radius_search
    visited = set()
    clusters = []

    for i in range(len(coords)):
        if i in visited:
            continue
        # 新開一個 cluster
        queue = [i]
        visited.add(i)
        current_cluster = [i]

        while queue:
            cur_idx = queue.pop()
            cur_pt = coords[cur_idx]
            # 在 KD-Tree 裡找所有與 cur_pt 距離 <= r 的點
            neighbors = []
            kdtree_radius_search(kdtree_root, cur_pt, objGDC.r, neighbors)
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    current_cluster.append(nb)

        clusters.append(current_cluster)

    points_px: List[Tuple[int, int]] = []
    for cluster in clusters:
        # 計算此 cluster 的中心 (平均值)
        xs_cluster = [coords[idx][0] for idx in cluster]
        ys_cluster = [coords[idx][1] for idx in cluster]
        mean_x = int(sum(xs_cluster) / len(xs_cluster))
        mean_y = int(sum(ys_cluster) / len(ys_cluster))

        points_px.append((mean_x, mean_y))

    return points_px

def group_and_draw_circles_MAIN(ptls:List[Tuple[int, int]],orig_w,orig_h, img: np.ndarray, x_pct: float, y_pct: float, r: int) -> np.ndarray:
    
    if algorithmSelect == 1:

        # ========================================方法:執行時間為 0.190812 秒=======================================
        # step 1 downsample 480
        OBJ_group_and_draw_circles = downsample480andMaskSide(ptls, orig_w, orig_h, img, x_pct, y_pct, r,True)
        # step2 
        ptlsHuman = masterSlow_group_and_draw_circles(OBJ_group_and_draw_circles, orig_w, orig_h)
        # step 999 draw and out
        return step3drawOut(
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            r,
            orig_w,
            orig_h, 
            ptlsHuman,
            True
        )

    if algorithmSelect == 2:

        # ========================================方法:執行時間為 0.005927 秒=======================================
        # step 1 downsample 480
        OBJ_group_and_draw_circles = downsample480andMaskSide(ptls, orig_w, orig_h, img, x_pct, y_pct, r,False)
        # step2 
        ptlsHuman = group_and_draw_circles_fastCV(OBJ_group_and_draw_circles)
        # 建立全黑影像 (3通道RGB，每個像素值為0)
        white_image = np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 0
        # step 999 draw and out
        return step3drawOut(
            white_image,
            r,
            orig_w,
            orig_h, 
            scale_points_to_target(
                ptlsHuman,
                OBJ_group_and_draw_circles.target_w, OBJ_group_and_draw_circles.target_h,
                orig_w, orig_h, 
                ),
            False
        )
    
    if algorithmSelect == 3:

        # ========================================方法:執行時間為 0.006328 秒=======================================
        # step 1 downsample 480
        OBJ_group_and_draw_circles = downsample480andMaskSide(ptls, orig_w, orig_h, img, x_pct, y_pct, r,False)
        # step2 
        ptlsHuman = group_and_draw_circles_KD(OBJ_group_and_draw_circles)
        # 建立全黑影像 (3通道RGB，每個像素值為0)
        white_image = np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 0
        # step 999 draw and out
        return step3drawOut(
            white_image,
            r,
            orig_w,
            orig_h, 
            scale_points_to_target(
                ptlsHuman,
                OBJ_group_and_draw_circles.target_w, OBJ_group_and_draw_circles.target_h,
                orig_w, orig_h, 
                ),
            False
        )

def group_and_draw_circles_fastCV(objGDC:STRUCT_group_and_draw_circles) -> List[Tuple[int, int]]:
    """
    1. 先建立 mask_full（與原本邏輯相同）； 
    2. 用手動畫圓的方式，把 mask_full 中每一個 True 的點，
       畫成半徑為 r 的實心圓到一張新的二值圖 dilated_manual；
    3. 對 dilated_manual 用 connectedComponentsWithStats 找出每個填充後大圓區塊
       的標籤、質心（centroids）等資訊；
    4. 最後把質心位置再畫一次半徑為 r 的圓到輸出影像上。
    """
    if not objGDC.ptls:
        return []
    
    # --- 2. 手動「畫大圓」到一張新的二值圖上（取代 getStructuringElement + dilate） ---
    # 建立一張全 0 的二值圖，大小同 mask_full
    #dilated_manual = np.zeros_like(objGDC.img, dtype=np.uint8)
    # 建立全黑影像 (3通道RGB，每個像素值為0)
    white_image = np.ones((objGDC.target_h, objGDC.target_w), dtype=np.uint8) * 0

    for (x0,y0) in objGDC.ptls:
        # 在 dilated_manual 上畫一個「白色實心圓」，半徑 r
        cv2.circle(white_image, (x0, y0), objGDC.r, 255, -1)

    # --- 3. 用 connectedComponentsWithStats 找每個「大圓群」的質心与統計資訊 ---
    # 這裡一定要先轉成 0/1 二值圖 (uint8)，connectedComponentsWithStats 會把 255 當成前景
    # connectivity=8 可以讓對角線也算是連通
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_image, connectivity=8)

    # 跳過第 0 個背景的 centroid，從 index 1 開始
    centroids_list: List[Tuple[int, int]] = [tuple(map(int, c)) for c in centroids[1:]]
    return centroids_list


def step3drawOut(img,r,w,h,ptls:List[Tuple[int, int]],doBlackWhiteConvert) -> np.ndarray:
    send_websocket_message("127.0.0.1", 9980, "/", "set"+json.dumps(normalize_points(ptls,w,h)))
    # 注意：此處用的座標、半徑都是原始解析度的單位
    fill_config = -1 if human_circle_fill else 2
    for x, y in flip_coordinates_batch(ptls,w,h):    
        cv2.circle(
            img,
            (x,y),
            r,
            _get_color_bgr("black") if doBlackWhiteConvert else _get_color_bgr("white"),
            fill_config,
        )

    if doBlackWhiteConvert:
        return process_image_white_BG_2_black_BG(img)
    else:
        return img

def masterSlow_group_and_draw_circles(OBJ_group_and_draw_circles: STRUCT_group_and_draw_circles,orig_w,orig_h)->List[Tuple[int, int]]:
    points_px: List[Tuple[int, int]] = []
    # 3) 用縮小後的 mask_small 計算 connectedComponents
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * OBJ_group_and_draw_circles.r + 1, 2 * OBJ_group_and_draw_circles.r + 1)
    )
    labels_small = cv2.connectedComponents(cv2.dilate(OBJ_group_and_draw_circles.img, kernel_small))[1]

    # ================================================================
    # 把 labels_small 放大回原始大小，方便後續找中心點
    # ================================================================
    # 先把 dtype 轉成 float32，再 resize 回原大小，用 最近鄰插值
    labels_up = cv2.resize(
        labels_small.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    # 接下來就把 labels_up 當作原本的 labels 來用
    labels = labels_up

    # ───────────────────────────────────────────────────────────────
    for lab in range(1, labels.max() + 1):
        ys, xs = np.where((labels == lab) & OBJ_group_and_draw_circles.maskFull)
        if xs.size:
            points_px.append((int(xs.mean()), int(ys.mean())))
    return points_px
            
def downsample480andMaskSide(ptls:List[Tuple[int, int]],w,h, img: np.ndarray, x_pct: float, y_pct: float, r: int,do_pre_img_pros) -> STRUCT_group_and_draw_circles:
    target_h = 480
    # 計算等比例縮放後的寬度
    scale = target_h / h
    target_w = int(w * scale)
    
    # 2) radius r 也要隨尺度縮放
    #    不可小於 1，否則 cv2.getStructuringElement 會當掉
    r_small = max(1, int(round(r * scale)))
    
    if do_pre_img_pros:

        dx, dy = int(w * x_pct / 100), int(h * y_pct / 100)
        mask = np.any(img != 255, axis=2)
        mask_crop = mask[dy : h - dy, dx : w - dx]
        mask_full = np.zeros_like(mask)
        mask_full[dy : h - dy, dx : w - dx] = mask_crop

        # ================================================================
        # 在這裡將 mask_full 的解析度先降到目標高度 480
        # ================================================================
        orig_h, orig_w = mask_full.shape  # 原始大小

        # 1) 從 bool -> uint8，再 resize（使用最近鄰最好保留二值）
        mask_small = cv2.resize(
            mask_full.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        return STRUCT_group_and_draw_circles(
            img=mask_small,
            r=r_small,
            ptls=[],#scale_points_to_target(filter_points(ptls,w,h,x_pct,y_pct), w, h, target_w, target_h),
            maskFull=mask_full,
            target_h=target_h,
            target_w=target_w,
        )
    
    else:

        return STRUCT_group_and_draw_circles(
            img=np.ones((2, 2), dtype=np.uint8),#mask_small,
            r=r_small,
            ptls=scale_points_to_target(filter_points(ptls,w,h,x_pct,y_pct), w, h, target_w, target_h),
            maskFull=np.ones((2, 2), dtype=np.uint8),#mask_full,
            target_h=target_h,
            target_w=target_w,
        )
    

def _process_sensor_data(radii_frame, angles_frame, sensor_trans):
    sensor_coords = []
    for s in range(4):
        if not radii_frame[s]:
            continue
        tx, ty = sensor_trans[s]
        ptsx, ptsy = [], []
        for r, ang_deg in zip(radii_frame[s], angles_frame[s]):
            if r > 15.0:
                continue
            a = math.radians(ang_deg)
            ptsx.append(r * math.sin(a) + tx)
            ptsy.append(r * math.cos(a) + ty)
        sensor_coords.append((ptsx, ptsy))
    return sensor_coords

def _get_color_bgr(color_name):
    """
    Convert color name to BGR tuple for OpenCV.
    """
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'purple': (128, 0, 128),
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    return color_map.get(color_name, (0, 0, 0))  # Default to black
def _world_to_pixel(x_world, y_world, canvas_w_px, canvas_h_px, plot_x_half, plot_y_half):
    """
    Convert world coordinates to pixel coordinates.
    """
    # Normalize world coordinates to [0, 1]
    x_norm = (x_world + plot_x_half) / (2 * plot_x_half)
    y_norm = (y_world + plot_y_half) / (2 * plot_y_half)
    
    # Convert to pixel coordinates (note: y is flipped for image coordinates)
    x_pixel = int(x_norm * canvas_w_px)
    y_pixel = int((1 - y_norm) * canvas_h_px)  # Flip y-axis
    
    # Clamp to valid pixel range
    x_pixel = max(0, min(canvas_w_px - 1, x_pixel))
    y_pixel = max(0, min(canvas_h_px - 1, y_pixel))
    
    return x_pixel, y_pixel

def frame2opencvIMG(frame_radii_data, frame_angles_data,
                    canvas_w_px, canvas_h_px,
                    plot_x_half, plot_y_half,
                    sensor_trans, colors_sensor,
                    fixed_dpi):
    image = np.full((canvas_h_px, canvas_w_px, 3), 255, dtype=np.uint8)
    points_px: List[Tuple[int, int]] = []
    # Process sensor data to get coordinates
    sensor_coords = _process_sensor_data(frame_radii_data, frame_angles_data, sensor_trans)
    
    # Draw points for each sensor
    for sensor_idx, (x_coords, y_coords) in enumerate(sensor_coords):
        if x_coords and y_coords:
            color_bgr = _get_color_bgr(colors_sensor[sensor_idx])
            
            for x_world, y_world in zip(x_coords, y_coords):
                x_pixel, y_pixel = _world_to_pixel(x_world, y_world, canvas_w_px, canvas_h_px, plot_x_half, plot_y_half)
                points_px.append((x_pixel, y_pixel))
                # Draw a small circle for each point (radius=1 for small dots)
                cv2.circle(image, (x_pixel, y_pixel), 1, color_bgr, -1)
    
    return group_and_draw_circles_MAIN(points_px,canvas_w_px,canvas_h_px, image, 5.0, 10.0, 55)
    # ==============這裡是參數，很重要，55那個值如果調成20會顯示雙腳，55會是人的肚子的腰圍=========================
    # 5.0、5.0 是 邊界 margin 的百分比



# ───────────────────  MAIN COOK  ────────────────────────────────
def onCook(scriptOp):
    chop = op('script_chop1') if bool_use_virtual_data else op('hokuyo1')
    if bool_use_virtual_data:
        r_vals_1 = chop['radius1'].vals  # numpy array	
        r_vals_2 = chop['radius2'].vals
        r_vals_3 = chop['radius3'].vals
        r_vals_4 = chop['radius4'].vals
        a_vals_1 = chop['angle1'].vals
        a_vals_2  = chop['angle2'].vals
        a_vals_3  = chop['angle3'].vals
        a_vals_4  = chop['angle4'].vals
    else:
        r_vals_1 = chop['radius'].vals
        a_vals_1 = chop['angle'].vals
        chop = op('hokuyo2')
        r_vals_2 = chop['radius'].vals
        a_vals_2 = chop['angle'].vals
        chop = op('hokuyo3')
        r_vals_3 = chop['radius'].vals
        a_vals_3 = chop['angle'].vals
        chop = op('hokuyo4')
        r_vals_4 = chop['radius'].vals
        a_vals_4 = chop['angle'].vals

    # 2️⃣  把四組資料組成函式需要的 list
    radii_frame  = [r_vals_1, r_vals_2, r_vals_3, r_vals_4]
    angles_frame = [a_vals_1, a_vals_2, a_vals_3, a_vals_4]

    # 3️⃣  固定參數
    W, H   = 1920,1080#1280, 720
    DPI    = 100
    plot_x_half = 6.7
    plot_y_half = 6.7 * H / W
    #+2
    sensor_trans = [(-6.7, -1.5), (6.7, 1.2),
                    (6.7, -1.5), (-6.7, 1.2)]
    colors_sensor = ['red', 'green', 'blue', 'purple']

    # 4️⃣  產生影像→傳給 Script TOP
    img_bgr = frame2opencvIMG(
        radii_frame, angles_frame,
        W, H, plot_x_half, plot_y_half,
        sensor_trans, colors_sensor, DPI
    )

    if bool_performance_measure:
    # 使用 lambda 版本的 timeit，測一輪呼叫 frame2opencvIMG 的花費時間
        execution_time = timeit.timeit(
            lambda: frame2opencvIMG(
                radii_frame, angles_frame,
                W, H, plot_x_half, plot_y_half,
                sensor_trans, colors_sensor, DPI
            ),
            number=1
        )

        print(f"執行時間為 {execution_time:.6f} 秒")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scriptOp.copyNumpyArray(img_rgb)
    return
